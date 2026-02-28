import pandas as pd
import glob
import os
import numpy as np

# load all trajectories.txt files & merge
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

# added lane change intent column to the data
def label_lane_changes(group, horizon=50):
    lanes = group['Lane_ID'].values
    n = len(lanes)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        current_lane = lanes[i]
        
        # Define the search window (from now up to 5s in the future)
        # We use i+1 to ensure we are looking at the "future"
        future_window = lanes[i+1 : i + horizon + 1]
        
        if len(future_window) == 0:
            labels[i] = 0
            continue
            
        # Find the first lane in the future window that is different from current
        # This captures the 'intent' the moment it enters the 5s horizon
        diff_lanes = future_window[future_window != current_lane]
        
        if len(diff_lanes) > 0:
            first_future_lane = diff_lanes[0]
            if first_future_lane < current_lane:
                labels[i] = 1  # Anticipating Left Change
            else:
                labels[i] = 2  # Anticipating Right Change
        else:
            labels[i] = 0  # No change in the next 5s
            
    group['lane_change'] = labels
    return group

# preprocess all data files and add lane change labels
def preprocess_all(data_folder="data", horizon=50):

    # load and merge all input txt files
    df = load_all_ngsim_txt(data_folder)

    # apply lane change labels per vehicle group
    df = df.groupby('Vehicle_Global_ID', group_keys=False).apply(label_lane_changes)

    # Quick check on the distribution
    print(df['lane_change'].value_counts())
    
    return df
