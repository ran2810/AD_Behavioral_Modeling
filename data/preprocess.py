import pandas as pd
import glob
import os

def load_all_ngsim_txt(data_folder):
    
    pattern = os.path.join(data_folder, '**', '*.txt')
    txt_files = glob.glob(pattern, recursive=True)
    all_dfs = []

    vehicle_offset = 0
    frame_offset = 0

    for f in txt_files:
        #print("reading file", f)
        df = pd.read_csv(f, sep=r"\s+", header=None, 
                         names=["Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y", "Global_X", 
                                "Global_Y", "v_len", "v_width", "v_class", "v_vel", "v_acc", "Lane_ID", "Preceeding", 
                                "Following", "Space_Headway", "Time_Headway"])

        # Create global IDs by adding max value as offset
        df["Vehicle_Global_ID"] = df["Vehicle_ID"] + vehicle_offset
        df["Frame_Global_ID"] = df["Frame_ID"] + frame_offset

        # Store original max values for next file
        vehicle_offset += df["Vehicle_ID"].max()
        frame_offset += df["Frame_ID"].max()
        #print("vehicle_offset", vehicle_offset)
        #print("frame_offset", frame_offset)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

def label_lane_changes(group, horizon=50):
    
    # Look ahead 50 frames within this vehicle's trajectory
    group['Future_Lane'] = group['Lane_ID'].shift(-horizon)
    
    # Initialize as 0 (No change)
    group['lane_change'] = 0
    
    # Filter for valid future look-aheads (ignore last 5s of vehicle life)
    mask = group['Future_Lane'].notna()
    
    # Lane ID assignment: Lane 1 is farthest left lane; lane 6 is farthest 
    # right lane. Lane 7 is the on-ramp at Powell Street, and Lane 9 is the shoulder on the right-side. )
    # Change Left: Future Lane ID is smaller than current
    group.loc[mask & (group['Future_Lane'] < group['Lane_ID']), 'lane_change'] = 1
    
    # Change Right: Future Lane ID is larger than current
    group.loc[mask & (group['Future_Lane'] > group['Lane_ID']), 'lane_change'] = 2
    
    return group

def preprocess_all(data_folder="data", horizon=50):
    df = load_all_ngsim_txt(data_folder)
    # Apply the labeling per vehicle
    df = df.groupby('Vehicle_Global_ID', group_keys=False).apply(label_lane_changes)

    # Drop the helper column
    df = df.drop(columns=['Future_Lane'])

    # Quick check on the distribution
    print(df['lane_change'].value_counts())
    return df
