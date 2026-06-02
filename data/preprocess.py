import gc
import glob
import os
import numpy as np
import pandas as pd

# Explicit dtypes keep each file at ~130 MB instead of ~260 MB (float32 vs float64)
_NGSIM_DTYPES = {
    'Vehicle_ID':    np.int32,
    'Frame_ID':      np.int32,
    'Total_Frames':  np.int32,
    'Global_Time':   np.int64,
    'Local_X':       np.float32,
    'Local_Y':       np.float32,
    'Global_X':      np.float32,
    'Global_Y':      np.float32,
    'v_len':         np.float32,
    'v_width':       np.float32,
    'v_class':       np.int8,
    'v_vel':         np.float32,
    'v_acc':         np.float32,
    'Lane_ID':       np.int8,
    'Preceeding':    np.int32,
    'Following':     np.int32,
    'Space_Headway': np.float32,
    'Time_Headway':  np.float32,
}

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

    # each file covers a different time window; offsets ensure IDs are globally unique
    for f in txt_files:
        print("reading file", f)
        col_names = list(_NGSIM_DTYPES.keys())
        df = pd.read_csv(f, sep=r"\s+", header=None,
                         names=col_names, dtype=_NGSIM_DTYPES)

        # add offset so Vehicle_ID 1 in file A != Vehicle_ID 1 in file B
        df["Vehicle_Global_ID"] = df["Vehicle_ID"] + vehicle_offset
        df["Frame_Global_ID"] = df["Frame_ID"] + frame_offset

        # tag source location — used later for lane topology rules
        if 'US-101' in f:
            df['Location'] = 'US-101'
        elif 'I-80' in f:
            df['Location'] = 'I-80'
        else:
            print("File is not part of the I-80/US-101 data", f)

        # shift offsets by the max in this file before reading the next
        vehicle_offset += df["Vehicle_ID"].max()
        frame_offset += df["Frame_ID"].max()
        #print("vehicle_offset", vehicle_offset)
        #print("frame_offset", frame_offset)
        all_dfs.append(df)
        del df
        gc.collect()

    combined = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()
    return combined


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

    # label every frame based on what the vehicle will do in the next `horizon` frames
    for i in range(n):
        current_lane = lanes[i]

        # look ahead from i+1 so we never include the current frame in the "future"
        future_window = lanes[i+1 : i + horizon + 1]

        # last few frames of a trajectory have no future — label as No Change
        if len(future_window) == 0:
            labels[i] = 0
            continue

        # grab only the frames where the lane actually changed
        diff_lanes = future_window[future_window != current_lane]

        if len(diff_lanes) > 0:
            # use the first different lane to determine direction
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
    # Preserve Vehicle_Global_ID before apply — newer pandas versions drop the groupby key
    vgid = df['Vehicle_Global_ID'].copy()
    frame_gid = df['Frame_Global_ID'].copy()
    df = df.groupby('Vehicle_Global_ID', group_keys=False).apply(label_lane_changes)
    if 'Vehicle_Global_ID' not in df.columns:
        df['Vehicle_Global_ID'] = vgid
    if 'Frame_Global_ID' not in df.columns:
        df['Frame_Global_ID'] = frame_gid

    # reset lane_change labels for on-ramp -> aux -> off-ramp
    # enabling just is leading to performance degradation
    #df = filter_ramp_transitions(df)

    # apply ground lane changes rules 
    df = apply_lane_change_rules(df)

    # Quick check on the distribution
    print(df['lane_change'].value_counts())
    
    return df
