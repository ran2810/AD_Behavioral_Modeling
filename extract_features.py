import gc
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


#########################################################################
# Kalman Smoothing
#########################################################################

def apply_kalman_smoothing(df, col_name='v_lat'):
    """
    Applies a 1D Kalman Filter to smooth lateral velocity (v_lat).
    """
    df[col_name] = df[col_name].fillna(0)
    smoothed_results = pd.Series(index=df.index, dtype=float)

    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State Transition Matrix (Constant Velocity Model): x = x + v*dt
    dt = 0.1  # NGSIM 10Hz
    kf.F = np.array([[1., dt],
                     [0., 1.]])

    # Measurement Function (We only measure velocity)
    kf.H = np.array([[0., 1.]])

    kf.P *= 10.   # Initial uncertainty
    kf.R = 1.0    # Measurement noise
    kf.Q = 0.01   # Process noise

    # process each vehicle separately — avoids filter state bleeding across different cars
    for vid, group in df.groupby('Vehicle_Global_ID'):
        v_data = group[col_name].values

        # reset filter state at the start of each vehicle's trajectory
        initial_v = np.clip(v_data[0], -1.0, 1.0)
        kf.x = np.array([[0.], [initial_v]])

        group_smoothed = []
        for z in v_data:
            # predict step: project state forward one time step
            kf.predict()
            # update step: correct with clamped measurement (spikes > 1.0 m/s are noise)
            kf.update(np.clip(z, -1.0, 1.0))
            group_smoothed.append(kf.x[1, 0])

        smoothed_results.loc[group.index] = group_smoothed

    return smoothed_results


#########################################################################
# Shared Feature Helpers
#########################################################################

def add_interaction_dynamics(df, window=10):
    """Gap rate of change and closing trend (window = frames per second)."""
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])

    # Negative = closing in; Positive = pulling away
    df['gap_rate_of_change'] = df.groupby('Vehicle_Global_ID')['actual_gap'].diff(periods=window)
    df['traffic_pressure'] = df['v_vel'] / (df['actual_gap'] + 1.0)
    df['gap_rate_of_change'] = df['gap_rate_of_change'].fillna(0)

    # Is the gap shrinking steadily over 1 second?
    df['gap_rate_trend_1s'] = df.groupby('Vehicle_Global_ID')['gap_rate_of_change'].transform(
        lambda x: x.rolling(window).mean()).fillna(0)

    return df


def add_displacement_features(df, fps=10):
    """Lateral displacement, acceleration variance, and persistence features."""
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    dt = 1.0 / fps
    window_1s = fps           # frames in 1 second
    window_15f = int(fps * 1.5)  # frames in 1.5 seconds

    # Lateral displacement over 1.5 seconds
    df['lat_dist_moved_15f'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
        lambda x: x.rolling(window_15f).sum() * dt
    ).fillna(0)

    df['v_lat_accel'] = df.groupby('Vehicle_Global_ID')['v_lat'].diff().fillna(0)
    df['a_lat'] = df.groupby('Vehicle_Global_ID')['v_lat'].diff().fillna(0) / dt

    # Lateral movement persistence over 1 second
    df['lat_displacement_1s'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
        lambda x: x.rolling(window_1s).sum() * dt).fillna(0)

    # Longitudinal acceleration variance over 1 second
    df['a_long_std_1s'] = df.groupby('Vehicle_Global_ID')['a_long'].transform(
        lambda x: x.rolling(window_1s).std()).fillna(0)

    return df


def add_lag_features(df, lag_05s=5, lag_1s=10, lag_2s=20):
    """Lateral velocity lags at 0.5s, 1.0s, and 2.0s back."""
    df['v_lat_lag_5']  = df.groupby('Vehicle_Global_ID')['v_lat'].shift(lag_05s).fillna(0)
    df['v_lat_lag_10'] = df.groupby('Vehicle_Global_ID')['v_lat'].shift(lag_1s).fillna(0)
    df['v_lat_lag_20'] = df.groupby('Vehicle_Global_ID')['v_lat'].shift(lag_2s).fillna(0)
    return df


#########################################################################
# NGSIM-specific: derive v_lat, a_long, lead-vehicle gap/speed
#########################################################################

def _add_ngsim_kinematics(df):
    """Derive v_lat and a_long from raw NGSIM position/velocity columns."""
    dt = 0.1
    df['v_lat'] = df.groupby('Vehicle_Global_ID')['Local_X'].diff() / dt
    df['a_long'] = df.groupby('Vehicle_Global_ID')['v_vel'].diff() / dt
    df['v_lat'] = df['v_lat'].fillna(0)
    df['a_long'] = df['a_long'].fillna(0)
    return df


def _add_ngsim_lead_features(df):
    """Merge lead vehicle velocity/position to compute actual_gap, rel_speed, TTC."""
    lead_info = df[['Frame_Global_ID', 'Vehicle_Global_ID', 'v_vel', 'Local_Y', 'v_len']].copy()
    lead_info = lead_info.rename(columns={
        'Vehicle_Global_ID': 'Preceeding',
        'v_vel': 'v_vel_lead',
        'Local_Y': 'Local_Y_lead',
        'v_len': 'v_len_lead'
    })

    df = pd.merge(df, lead_info, on=['Frame_Global_ID', 'Preceeding'], how='left')
    del lead_info
    gc.collect()

    df['rel_speed'] = (df['v_vel_lead'] - df['v_vel']).fillna(0)

    # NGSIM Space_Headway is center-to-center; subtract half lengths to get gap
    df['actual_gap'] = (
        (df['Local_Y_lead'] - df['Local_Y']) - (df['v_len_lead'] / 2 + df['v_len'] / 2)
    ).fillna(1000)

    closing_vel = df['v_vel'] - df['v_vel_lead']
    df['TTC'] = np.where(
        (closing_vel > 0) & (df['actual_gap'] > 0),
        df['actual_gap'] / closing_vel,
        100
    )
    df['TTC'] = df['TTC'].clip(upper=100)

    df = df.drop(columns=['v_vel_lead', 'Local_Y_lead', 'v_len_lead'], errors='ignore')
    return df


def _convert_ngsim_to_si(df):
    """Convert NGSIM columns from feet to meters."""
    FT_TO_M = 0.3048
    cols_to_scale = [
        'Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_len', 'v_width', 'Space_Headway', 'v_acc',
        'actual_gap', 'lat_displacement_1s', 'gap_rate_trend_1s', 'gap_rate_of_change',
        'v_vel', 'v_lat', 'rel_speed', 'v_lat_lag_5', 'v_lat_lag_10', 'v_lat_accel', 'lat_dist_moved_15f',
        'a_lat', 'a_long', 'a_long_std_1s'
    ]
    existing_cols = [c for c in cols_to_scale if c in df.columns]
    df[existing_cols] = df[existing_cols] * FT_TO_M
    # Downcast back to float32 after scaling (multiplication promotes to float64)
    df[existing_cols] = df[existing_cols].astype(np.float32)
    return df


#########################################################################
# HighD-specific: derive actual_gap, rel_speed, TTC from existing columns
#########################################################################

def _add_highd_lead_features(df):
    """
    HighD provides dhw (Space_Headway) and TTC_raw directly.
    Compute actual_gap and rel_speed.
    """
    # actual_gap: dhw is center-to-center, subtract front vehicle half-length
    df['actual_gap'] = (df['Space_Headway'] - (df['v_len'] / 2)).fillna(1000).clip(lower=0)
    df.loc[df['Space_Headway'] == 0, 'actual_gap'] = 1000  # no lead vehicle

    # TTC already cleaned in preprocess_highd (0 → 100)
    df['TTC'] = df['TTC_raw'].fillna(100).clip(upper=100)

    # rel_speed: merge preceding vehicle's v_vel
    lead_info = df[['Frame_Global_ID', 'Vehicle_Global_ID', 'v_vel']].copy()
    lead_info = lead_info.rename(columns={
        'Vehicle_Global_ID': 'Preceeding',
        'v_vel': 'v_vel_lead'
    })
    df = pd.merge(df, lead_info, on=['Frame_Global_ID', 'Preceeding'], how='left')
    df['rel_speed'] = (df['v_vel_lead'] - df['v_vel']).fillna(0)
    df = df.drop(columns=['v_vel_lead'], errors='ignore')

    return df


#########################################################################
# Main Entry Point
#########################################################################

def calculate_features(df, dataset='ngsim'):
    """
    Perform feature engineering for the given dataset.

    Args:
        df:      DataFrame from preprocess_all() or preprocess_highd()
        dataset: 'ngsim' or 'highd'

    Returns:
        df with all model features added.
    """
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)

    if dataset == 'ngsim':
        fps = 10  # NGSIM recorded at 10 Hz

        print("Calculating NGSIM kinematics...")
        df = _add_ngsim_kinematics(df)

        print("Calculating NGSIM lead-vehicle features...")
        df = _add_ngsim_lead_features(df)

        df = add_displacement_features(df, fps=fps)
        df = add_interaction_dynamics(df, window=fps)
        df = add_lag_features(df, lag_05s=5, lag_1s=10)

        print("Converting to SI units...")
        df = _convert_ngsim_to_si(df)

    elif dataset == 'highd':
        fps = 25  # HighD recorded at 25 Hz

        print("Calculating HighD lead-vehicle features...")
        # v_lat and a_long already normalized in preprocess_highd
        df = _add_highd_lead_features(df)
        df = add_displacement_features(df, fps=fps)
        df = add_interaction_dynamics(df, window=fps)
        df = add_lag_features(df, lag_05s=12, lag_1s=25, lag_2s=50)
        # No unit conversion — HighD is already SI

    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Use 'ngsim' or 'highd'.")

    print(f"Feature engineering done [{dataset}]!")
    return df
