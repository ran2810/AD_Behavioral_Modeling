import pandas as pd
import glob
import os
import numpy as np


def load_all_highd_csv(data_folder):
    """
    Load all XX_tracks.csv and XX_tracksMeta.csv files from a highD data folder.
    HighD recordings are numbered 01-60, each with separate tracks and meta files.
    All values are already in SI units (meters, m/s, m/s^2).
    """
    track_files = sorted(glob.glob(os.path.join(data_folder, '*_tracks.csv')))
    meta_files = sorted(glob.glob(os.path.join(data_folder, '*_tracksMeta.csv')))

    if not track_files:
        raise FileNotFoundError(f"No HighD track files found in: {data_folder}")

    # Build a lookup from tracksMeta: id -> (width, height, drivingDirection, class)
    meta_lookup = {}
    for mf in meta_files:
        meta_df = pd.read_csv(mf)
        recording_id = int(os.path.basename(mf).split('_')[0])
        for _, row in meta_df.iterrows():
            meta_lookup[(recording_id, int(row['id']))] = {
                'v_len': row['width'],     # HighD 'width' = vehicle length (driving direction)
                'v_width': row['height'],  # HighD 'height' = vehicle width (lateral)
                'drivingDirection': int(row['drivingDirection']),
                'v_class': 1 if row['class'] == 'Car' else 2
            }

    all_dfs = []
    vehicle_offset = 0
    frame_offset = 0

    for tf in track_files:
        recording_id = int(os.path.basename(tf).split('_')[0])
        print(f"Reading HighD recording {recording_id:02d}: {tf}")
        df = pd.read_csv(tf)

        # Rename to shared column names
        df = df.rename(columns={
            'id':             'Vehicle_ID',
            'frame':          'Frame_ID',
            'laneId':         'Lane_ID',
            'x':              'Local_Y',   # x = driving direction -> maps to NGSIM Local_Y
            'y':              'Local_X',   # y = lateral direction -> maps to NGSIM Local_X
            'xVelocity':      'v_vel_raw', # signed: negative for direction=1
            'yVelocity':      'v_lat_raw', # lateral velocity (negative=left, positive=right)
            'xAcceleration':  'a_long_raw',
            'yAcceleration':  'a_lat_raw',
            'dhw':            'Space_Headway',
            'thw':            'Time_Headway',
            'ttc':            'TTC_raw',
            'precedingId':    'Preceeding',
            'followingId':    'Following',
            'precedingXVelocity': 'v_vel_lead_raw',
        })

        # Attach vehicle-level metadata (length, direction, class)
        df['v_len'] = df['Vehicle_ID'].map(
            lambda vid: meta_lookup.get((recording_id, vid), {}).get('v_len', 4.5))
        df['v_width'] = df['Vehicle_ID'].map(
            lambda vid: meta_lookup.get((recording_id, vid), {}).get('v_width', 2.0))
        df['drivingDirection'] = df['Vehicle_ID'].map(
            lambda vid: meta_lookup.get((recording_id, vid), {}).get('drivingDirection', 1))
        df['v_class'] = df['Vehicle_ID'].map(
            lambda vid: meta_lookup.get((recording_id, vid), {}).get('v_class', 1))

        # Normalize velocities so v_vel is always positive (speed in driving direction)
        # Direction=1: vehicles move left -> xVelocity is negative
        # Direction=2: vehicles move right -> xVelocity is positive
        df['v_vel'] = df['v_vel_raw'].abs()
        df['a_long'] = np.where(df['drivingDirection'] == 1, -df['a_long_raw'], df['a_long_raw'])

        # Lateral: yVelocity convention -> negative=left lane, positive=right lane
        # This aligns with NGSIM: v_lat negative = moving toward lane 1 (left)
        df['v_lat'] = df['v_lat_raw']

        # Global IDs to avoid collisions across recordings
        df['Vehicle_Global_ID'] = df['Vehicle_ID'] + vehicle_offset
        df['Frame_Global_ID'] = df['Frame_ID'] + frame_offset
        df['Location'] = f'HighD-{recording_id:02d}'
        df['Recording_ID'] = recording_id

        vehicle_offset += df['Vehicle_ID'].max()
        frame_offset += df['Frame_ID'].max()

        # Drop raw signed columns (keep cleaned versions)
        df = df.drop(columns=['v_vel_raw', 'a_long_raw', 'a_lat_raw', 'v_lat_raw', 'v_vel_lead_raw'], errors='ignore')

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def apply_lane_change_rules_highd(df):
    """
    Set can_go_left / can_go_right flags for HighD lanes.
    HighD laneId 1 = above top lane marking (boundary, not a real driving lane).
    The actual driving lanes for each direction are determined per recording.
    Direction=1: smaller laneId = left (fast), larger = right (slow)
    Direction=2: same convention (lower laneId = left in driving direction)
    """
    df['can_go_left'] = 1
    df['can_go_right'] = 1
    df['is_special_lane'] = 0

    for rec_id, rec_df in df.groupby('Recording_ID'):
        for direction in [1, 2]:
            mask = (df['Recording_ID'] == rec_id) & (df['drivingDirection'] == direction)
            if mask.sum() == 0:
                continue

            valid_lanes = sorted(df.loc[mask, 'Lane_ID'].unique())

            # Boundary lanes (laneId=1 and max) are not real driving lanes
            # but vehicles rarely appear there; just protect the edges
            min_lane = valid_lanes[0]
            max_lane = valid_lanes[-1]

            # Leftmost lane: can't go further left
            df.loc[mask & (df['Lane_ID'] == min_lane), 'can_go_left'] = 0
            # Rightmost lane: can't go further right
            df.loc[mask & (df['Lane_ID'] == max_lane), 'can_go_right'] = 0

    return df


def label_lane_changes_highd(group, horizon=50):
    """
    Label lane change intent for HighD vehicles.
    Uses the same logic as NGSIM: look up to 5s ahead (50 frames at 25Hz -> adjust if needed).
    For direction=1: smaller laneId = left lane (same as NGSIM convention).
    For direction=2: same convention -> lower laneId = left in driving direction.
    """
    lanes = group['Lane_ID'].values
    n = len(lanes)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        current_lane = lanes[i]
        future_window = lanes[i + 1: i + horizon + 1]

        if len(future_window) == 0:
            labels[i] = 0
            continue

        diff_lanes = future_window[future_window != current_lane]

        if len(diff_lanes) > 0:
            first_future_lane = diff_lanes[0]
            if first_future_lane < current_lane:
                labels[i] = 1  # Left lane change
            else:
                labels[i] = 2  # Right lane change
        else:
            labels[i] = 0  # No change

    group['lane_change'] = labels
    return group


def preprocess_highd(data_folder):
    """
    Load, label, and prepare HighD data for feature engineering.
    Returns a DataFrame with the same schema expected by calculate_features(dataset='highd').
    """
    highd_folder = os.path.join(data_folder, 'highD-dataset-v1.0\data')
    df = load_all_highd_csv(highd_folder)

    # Sort by vehicle and frame before labeling
    df = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)

    # HighD frame rate is 25Hz -> 5 seconds = 125 frames
    # Use horizon=125 for equivalent 5s lookahead as NGSIM (10Hz -> 50 frames)
    # Preserve global IDs — newer pandas versions drop the groupby key from apply result
    vgid = df['Vehicle_Global_ID'].copy()
    frame_gid = df['Frame_Global_ID'].copy()
    df = df.groupby('Vehicle_Global_ID', group_keys=False).apply(
        lambda g: label_lane_changes_highd(g, horizon=125)
    )
    if 'Vehicle_Global_ID' not in df.columns:
        df['Vehicle_Global_ID'] = vgid
    if 'Frame_Global_ID' not in df.columns:
        df['Frame_Global_ID'] = frame_gid

    df = apply_lane_change_rules_highd(df)

    # HighD TTC: 0 means no preceding vehicle (not actual 0 TTC)
    # Replace with 100 to match NGSIM convention
    if 'TTC_raw' in df.columns:
        df['TTC_raw'] = df['TTC_raw'].replace(0, 100).fillna(100)

    print(df['lane_change'].value_counts())
    return df
