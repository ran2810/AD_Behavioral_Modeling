import numpy as np
import pandas as pd

def add_interaction_dynamics(df):
    # Sort to ensure temporal consistency
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    
    # 1. Gap Rate of Change (delta_gap over 1 second)
    # Negative = closing in; Positive = pulling away
    df['gap_rate_of_change'] = df.groupby('Vehicle_Global_ID')['actual_gap'].diff(periods=10)
    
    # 2. Velocity-to-Gap Ratio
    # High speed + small gap = high pressure to change lanes
    # Adding 1 to denominator to avoid division by zero
    df['traffic_pressure'] = df['v_vel'] / (df['actual_gap'] + 1.0)
    
    # Handle NaNs from the diff/shift
    df['gap_rate_of_change'] = df['gap_rate_of_change'].fillna(0)
    
    return df


def add_displacement_features(df):
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    
    # 1. Cumulative Lateral Displacement (last 1.5 seconds / 15 frames)
    # This captures the "Net" movement.
    df['lat_dist_moved_15f'] = df.groupby('Vehicle_Global_ID')['v_lat'].transform(
        lambda x: x.rolling(15).sum() * 0.1 # Integrating velocity over 0.1s steps
    ).fillna(0)
    
    # 2. Lateral Velocity Trend
    # Is the car accelerating its sideway movement?
    df['v_lat_accel'] = df.groupby('Vehicle_Global_ID')['v_lat'].diff().fillna(0)
    
    return df

def calculate_features(df):
    # 1. Sort for consistent time-series calculations
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)
    dt = 0.1  # 100ms interval

    # --- Individual Vehicle Dynamics ---
    print("Calculating Individual Dynamics...")
    # Lateral Velocity: Change in Local_X over time
    df['v_lat'] = df.groupby('Vehicle_Global_ID')['Local_X'].diff() / dt
    
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
    # We need v_vel and Local_Y of the 'Preceeding' vehicle ID
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
    cols_to_scale = ['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_len', 'v_width', 'Space_Headway', 'actual_gap', # distances
                     'v_vel', 'v_lat', 'rel_speed',  # velocities
                     'v_acc', 'a_long'] # accelerations

    # Apply conversion only to columns that exist in the dataframe
    # Define conversion factor (feet to meters)
    FT_TO_M = 0.3048
    existing_cols = [c for c in cols_to_scale if c in df.columns]
    df[existing_cols] = df[existing_cols] * FT_TO_M

    print("Feature Engineering Done..!!")
    
    return df
