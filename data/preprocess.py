import pandas as pd
import glob
import os
import numpy as np

# load all trajectories.txt files & merge
def load_all_ngsim_txt(data_folder):
    """
    load all trajectories.txt files
    """
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

        # store location
        if 'US-101' in f:
            df['Location'] = 'US-101'
        elif 'I-80' in f:
            df['Location'] = 'I-80'
        else:
            print("File is not part of the I-80/US-101 data", f)

        # Store original max values for next file
        vehicle_offset += df["Vehicle_ID"].max()
        frame_offset += df["Frame_ID"].max()
        #print("vehicle_offset", vehicle_offset)
        #print("frame_offset", frame_offset)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def filter_ramp_transitions(df):
    """ handle Lane_ID change without actual Lane Change"""

    # Identify indices where a "lane change" is recorded
    # but it's just the ramp merging or the auxiliary exiting.
    
    # 1. US-101: 7 (On-ramp) -> 6 (Aux) is often just the lane joining.
    # 2. US-101: 6 (Aux) -> 8 (Off-ramp) is often just the lane splitting.
    
    # If current Lane is 7 and target is 6, labelled as left --> set lane_change to 0
    mask_7_to_6 = (df['Location'] == 'US-101') & (df['Lane_ID'] == 7) & (df['lane_change'] == 1)
    # If current Lane is 6 and target is 8, labelled as right --> set lane_change to 0
    mask_6_to_8 = (df['Location'] == 'US-101') & (df['Lane_ID'] == 6) & (df['lane_change'] == 2)
    
    # Apply the filter
    df.loc[mask_7_to_6, 'lane_change'] = 0
    df.loc[mask_6_to_8, 'lane_change'] = 0


    # # 1. I-80: 7 (On-ramp) -> 6 (Aux) is often just the lane joining.
    # # 2. I-80: 6 (Aux) -> 7 (On-ramp) is often just the lane merging.
    
    # If current Lane is 7 and target is 6, labelled as left --> set lane_change to 0
    #mask_7_to_6 = (df['Location'] == 'I-80') & (df['Lane_ID'] == 7) & (df['lane_change'] == 1)
    # If current Lane is 6 and target is 7, labelled as right --> set lane_change to 0
    #mask_6_to_7 = (df['Location'] == 'I-80') & (df['Lane_ID'] == 6) & (df['lane_change'] == 2)
    
    # # Apply the filter
    #df.loc[mask_7_to_6, 'lane_change'] = 0
    #df.loc[mask_6_to_7, 'lane_change'] = 0

    return df


def apply_lane_change_rules(df):
    """apply ground lane change rules"""

    # Initialize defaults
    df['can_go_left'] = 1
    df['can_go_right'] = 1
    df['is_special_lane'] = 0 # Ramps, auxiliary, or shoulders

    # --- US-101 Rules ---
    mask_101 = (df['Location'] == 'US-101')
    # Lane 1 is the fast lane (far left) --> cannot any further left
    df.loc[mask_101 & (df['Lane_ID'] == 1), 'can_go_left'] = 0
    # Lane 6 is the Aux lane. Lanes 7 & 8 are Ramps.
    df.loc[mask_101 & (df['Lane_ID'].isin([6, 7, 8])), 'is_special_lane'] = 1
    # Lane 8 is the OFF-RAMP: You physically cannot go right/left anymore
    df.loc[mask_101 & (df['Lane_ID'] == 8), 'can_go_right'] = 0
    df.loc[mask_101 & (df['Lane_ID'] == 8), 'can_go_left']  = 0
    # Lane 7 is the ON-RAMP: You must go left to merge, right is usually blocked
    df.loc[mask_101 & (df['Lane_ID'] == 7), 'can_go_right'] = 0

    # --- I-80 Rules ---
    mask_80 = (df['Location'] == 'I-80')
    # Lane 1 is the HOV/fast lane (far left)
    df.loc[mask_80 & (df['Lane_ID'] == 1), 'can_go_left'] = 0
    # Lane 6 is the far right mainline. Lane 7 is on-ramp
    df.loc[mask_80 & (df['Lane_ID'] == 7), 'is_special_lane'] = 1
    # Lane 6 can still go right into the Powell ramp for short length of 140ft -> see anaylsis .pdf
    df.loc[mask_80 & (df['Lane_ID'] == 6) & (df['Local_Y'] > 666), 'can_go_right'] = 0
    # Lane 7 (On-ramp): Must go only left to merge.
    df.loc[mask_80 & (df['Lane_ID'] == 7), 'can_go_right'] = 0

    return df


def label_lane_changes(group, horizon=50):
    """ add lane change intent column to the data for t secs before"""
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


def preprocess_all(data_folder="data"):
    """preprocess all data files and add lane change labels"""

    # load and merge all input txt files
    df = load_all_ngsim_txt(data_folder)

    # apply lane change labels per vehicle group
    df = df.groupby('Vehicle_Global_ID', group_keys=False).apply(label_lane_changes)

    # reset lane_change labels for on-ramp -> aux -> off-ramp
    # enabling just is leading to performance degradation
    #df = filter_ramp_transitions(df)

    # apply ground lane changes rules 
    df = apply_lane_change_rules(df)

    # Quick check on the distribution
    print(df['lane_change'].value_counts())
    
    return df
