import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

def apply_kalman_smoothing(df, col_name='v_lat'):
    """
    Applies a 1D Kalman Filter to smooth lateral velocity (v_lat).
    """
    # Initialize Kalman Filter
    # dim_x=2: State is [position, velocity]
    # dim_z=1: Measurement is just [velocity]

    df[col_name] = df[col_name].fillna(0)
    smoothed_results = pd.Series(index=df.index, dtype=float)
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    
    # State Transition Matrix (Constant Velocity Model)
    # x = x + v*dt
    dt = 0.1  # NGSIM 10Hz
    kf.F = np.array([[1., dt],
                     [0., 1.]])
    
    # Measurement Function (We only measure velocity)
    kf.H = np.array([[0., 1.]])
    
    kf.P *= 10.                           # Initial uncertainty
    kf.R = 1.0                           # Measurement noise R 5.0 is too high 
    kf.Q = 0.01                           # Process noise - Update velocity faster so 0.01
    
    # Apply per vehicle to avoid jumping between different cars
    for vid, group in df.groupby('Vehicle_Global_ID'):
        v_data = group[col_name].values
        
        # Reset filter for new vehicle
        initial_v = np.clip(v_data[0], -1.0, 1.0)
        kf.x = np.array([[0.], [initial_v]])
        
        group_smoothed = []
        for z in v_data:
            kf.predict()

            z_safe = np.clip(z, -1.0, 1.0)
            kf.update(z_safe)
            group_smoothed.append(kf.x[1, 0]) # Extract smoothed velocity
            
        # Map the list back to the specific indices of this vehicle
        smoothed_results.loc[group.index] = group_smoothed
        
    return smoothed_results

def add_interaction_dynamics(df):
    # Sort to ensure temporal consistency
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    
    # Gap Rate of Change (delta_gap over 1 second)
    # Negative = closing in; Positive = pulling away
    df['gap_rate_of_change'] = df.groupby('Vehicle_Global_ID')['actual_gap'].diff(periods=10)
    
    # Velocity-to-Gap Ratio
    # High speed + small gap = high pressure to change lanes
    # Adding 1 to denominator to avoid division by zero
    df['traffic_pressure'] = df['v_vel'] / (df['actual_gap'] + 1.0)
    
    # Handle NaNs from the diff/shift
    df['gap_rate_of_change'] = df['gap_rate_of_change'].fillna(0)

    # Gap Closing Trend (Is the gap shrinking steadily?)
    # If this is negative and stable, it's a high-intent signal
    df['gap_rate_trend_1s'] = df.groupby('Vehicle_Global_ID')['gap_rate_of_change'].transform(
        lambda x: x.rolling(10).mean()).fillna(0)
    
    return df


def add_displacement_features(df):
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    
    # Cumulative Lateral Displacement (last 1.5 seconds / 15 frames)
    # This captures the "Net" movement.
    df['lat_dist_moved_15f'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
        lambda x: x.rolling(15).sum() * 0.1 # Integrating velocity over 0.1s steps
    ).fillna(0)
    
    # Lateral Velocity Trend
    # Is the car accelerating its sideway movement?
    df['v_lat_accel'] = df.groupby('Vehicle_Global_ID')['v_lat'].diff().fillna(0)

    # --- EXISTING PHYSICS ---
    df['a_lat'] = df.groupby('Vehicle_Global_ID')['v_lat'].diff().fillna(0) / 0.1
    
    # Lateral Movement Persistence (Is the drift consistent?)
    # Summing v_lat over 1 second (10 frames)
    df['lat_displacement_1s'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
        lambda x: x.rolling(10).sum() * 0.1).fillna(0)
    
    # Acceleration Variance (Jerky driving vs Smooth maneuver)
    df['a_long_std_1s'] = df.groupby('Vehicle_Global_ID')['a_long'].transform(
        lambda x: x.rolling(10).std()).fillna(0)
    
    return df

def calculate_features(df):
    """ perform feature engineering for the model """

    # Sort for consistent time-series calculations
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)
    dt = 0.1  # 100ms interval

    # --- Individual Vehicle Dynamics ---
    print("Calculating Individual Dynamics...")
    # Lateral Velocity: Change in Local_X over time
    df['v_lat'] = df.groupby('Vehicle_Global_ID')['Local_X'].diff() / dt

    # # # spikes of +4.0; highway moves rarely exceed 1.0 m/s
    # df['v_lat'] = df['v_lat'].clip(-1.2, 1.2)
    
    # # (Median Filter) 
    # # This removes 'salt and pepper' noise without blurring the signal edges
    # df['v_lat'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
    #     lambda x: x.rolling(window=3, center=True).median()
    # ).fillna(df['v_lat'])

    #df['v_lat'] = apply_kalman_smoothing(df)

    
    # Longitudinal Acceleration (if not already provided or to verify)
    # Using v_vel (longitudinal velocity)
    df['a_long'] = df.groupby('Vehicle_Global_ID')['v_vel'].diff() / dt

    # Create the lag features per vehicle
    # 5 frames = 0.5s, 10 frames = 1.0s
    df['v_lat_lag_5'] = df.groupby('Vehicle_Global_ID')['v_lat'].shift(5)
    df['v_lat_lag_10'] = df.groupby('Vehicle_Global_ID')['v_lat'].shift(10)

    df = add_displacement_features(df)

    # --- Interactive Features (Lead Vehicle) ---
    print("Calculating Lead Vehicle Features...")
    
    # Create a mapping for quick lookup of lead vehicle stats per frame
    # need v_vel and Local_Y of the 'Preceeding' vehicle ID
    lead_info = df[['Frame_Global_ID', 'Vehicle_Global_ID', 'v_vel', 'Local_Y', 'v_len']].copy()
    lead_info = lead_info.rename(columns={
        'Vehicle_Global_ID': 'Preceeding',
        'v_vel': 'v_vel_lead',
        'Local_Y': 'Local_Y_lead',
        'v_len': 'v_len_lead'
    })

    # Merge lead vehicle info onto the main dataframe
    df = pd.merge(df, lead_info, on=['Frame_Global_ID', 'Preceeding'], how='left')

    # --- Feature Calculations ---
    
    # Relative Speed (v_lead - v_ego)
    # Positive means lead is pulling away, negative means closing in
    df['rel_speed'] = df['v_vel_lead'] - df['v_vel']

    # Space Headway (Actual Gap)
    # NGSIM 'Space_Headway' is center-to-center. 
    # Gap = (Y_lead - Y_ego) - (Len_lead/2 + Len_ego/2)
    df['actual_gap'] = (df['Local_Y_lead'] - df['Local_Y']) - (df['v_len_lead']/2 + df['v_len']/2)
    # Fill cases where there is no lead vehicle with a large constant
    df['actual_gap'] = df['actual_gap'].fillna(1000) 

    # add gap_rate_of_change
    df = add_interaction_dynamics(df)

    # Time-to-Collision (TTC)
    # Formula: Gap / Relative_Velocity (only if Relative_Velocity is negative, i.e., closing)
    # Relative Velocity here is (v_ego - v_lead)
    closing_vel = df['v_vel'] - df['v_vel_lead']
    
    df['TTC'] = np.where(
        (closing_vel > 0) & (df['actual_gap'] > 0),
        df['actual_gap'] / closing_vel,
        100 # Default value for no collision risk
    )
    
    # Cap TTC at 100 seconds to avoid infinities
    df['TTC'] = df['TTC'].clip(upper=100)

    # Clean up helper columns
    df = df.drop(columns=['v_vel_lead', 'Local_Y_lead', 'v_len_lead'])

    # Fill cases where there is no lead vehicle with 0
    df['v_lat_lag_5'] = df['v_lat_lag_5'].fillna(0)
    df['v_lat_lag_10'] =  df['v_lat_lag_10'].fillna(0)  
    df['rel_speed'] = df['rel_speed'].fillna(0) 
    df['v_lat'] = df['v_lat'].fillna(0) 
    df['a_long'] = df['a_long'].fillna(0) 

    # ----columns required for conversion from feet to metres---
    print("Converting to SI units....")
    cols_to_scale = ['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_len', 'v_width', 'Space_Headway', 'v_acc', 
                     'actual_gap', 'lat_displacement_1s', 'gap_rate_trend_1s', 'gap_rate_of_change',# distances
                     'v_vel', 'v_lat', 'rel_speed', 'v_lat_lag_5', 'v_lat_lag_10', 'v_lat_accel', 'lat_dist_moved_15f',  # velocities
                     'a_lat', 'a_long', 'a_long_std_1s'] # accelerations

    # Apply conversion only to columns that exist in the dataframe
    # Define conversion factor (feet to meters)
    FT_TO_M = 0.3048
    existing_cols = [c for c in cols_to_scale if c in df.columns]
    df[existing_cols] = df[existing_cols] * FT_TO_M

    print("Feature Engineering Done..!!")
    
    return df
