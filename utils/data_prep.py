import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def data_split_with_sampling_mining(df, features, sampling_keep_factor=4):
    """
    Splits data by Vehicle ID to prevent leakage and performs 
    Hard Negative Mining on the training set.
    """
    # 1. Split based on Vehicle_Global_ID to prevent temporal leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # We use the full df for splitting to keep track of indices correctly
    train_idx, test_idx = next(gss.split(df, groups=df['Vehicle_Global_ID']))
    
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 2. Identify Class 0 (No Change) vs. Maneuvers (1 & 2) in Training Set
    idx_maneuver = df_train[df_train['lane_change'] != 0].index
    idx_no_change = df_train[df_train['lane_change'] == 0].index

    # 3. Define "Hard" Negatives (Class 0 samples that look like Lane Changes)
    # These are frames where the car is swerving or closing gaps quickly
    # Thresholds: v_lat > 0.12 m/s OR Gap closing faster than 1.2 m/s
    is_hard_mask = (
        (df_train.loc[idx_no_change, 'v_lat'].abs() > 0.25) | 
        (df_train.loc[idx_no_change, 'gap_rate_of_change'] < -2.5)
    )
    
    idx_hard_no_change = idx_no_change[is_hard_mask]
    idx_easy_no_change = idx_no_change[~is_hard_mask]

    # 4. Calculate Sampling Budget
    # Target total Class 0 size based on your keep_factor
    num_maneuvers = len(idx_maneuver)
    total_no_change_budget = num_maneuvers * sampling_keep_factor
    
    # We keep ALL Hard Negatives first
    num_hard_to_keep = len(idx_hard_no_change)
    
    # Remaining budget for "Boring" (Easy) samples
    num_easy_to_sample = max(0, total_no_change_budget - num_hard_to_keep)

    # 5. Sample the "Easy" Negatives
    if num_easy_to_sample < len(idx_easy_no_change):
        sampled_easy_idx = np.random.choice(
            idx_easy_no_change, size=num_easy_to_sample, replace=False
        )
    else:
        sampled_easy_idx = idx_easy_no_change

    # 6. Combine Indices (Maneuvers + Hard Nones + Sampled Easy Nones)
    final_train_indices = np.concatenate([
        idx_maneuver, 
        idx_hard_no_change, 
        sampled_easy_idx
    ])

    # 7. Create Final Training and Test Sets
    X_train_sub = df_train.loc[final_train_indices, features].sample(frac=1, random_state=42)
    y_train_sub = df_train.loc[X_train_sub.index, 'lane_change']

    X_test = df_test[features]
    y_test = df_test['lane_change']

    # --- Print Summary ---
    print(f"Original Training Size: {len(df_train)}")
    print(f"Hard Mining Summary:")
    print(f" - Lane Changes Kept: {num_maneuvers}")
    print(f" - Hard 'None' Cases Kept: {num_hard_to_keep}")
    print(f" - Easy 'None' Cases Sampled: {len(sampled_easy_idx)}")
    print(f"Final Training Size: {len(y_train_sub)}")
    print("New Class Distribution:\n", y_train_sub.value_counts())
    
    return X_train_sub, X_test, y_train_sub, y_test

def data_split_with_sampling(df, features, sampling_keep_factor=4):


    # Split based on Veh_ID to prevent leakage 
    # single vehicle's trajectory is made of hundreds of consecutive frames, the model might see frame t in training and frame t+1 in testing 
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Groupby vehicle ID so that particular ID exists either in train or test data
    train_idx, test_idx = next(gss.split(df[features], df['lane_change'], groups=df['Vehicle_Global_ID']))
    
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx]['lane_change']

    X_test = df.iloc[test_idx][features]
    y_test = df.iloc[test_idx]['lane_change']

    # Train/test split
    # AVOIDED due to data leakage - split is done by y not by vehicle_ID. therefore the same vehicle ID data exists in train and test
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    # Identify indices for each class in the training set
    idx_no_change = y_train[y_train == 0].index
    idx_change_left = y_train[y_train == 1].index
    idx_change_right = y_train[y_train == 2].index

    # Count the rare events (Labels 1 and 2)
    num_lane_changes = len(idx_change_left) + len(idx_change_right)

    #  Apply sampling keep factor - data with no lane changes
    num_no_change_to_keep = num_lane_changes * sampling_keep_factor 

    # Randomly sample the 'No Change' indices
    num_no_change_to_keep = min(num_no_change_to_keep, len(idx_no_change))
    sampled_no_change_idx = np.random.choice(idx_no_change, size=num_no_change_to_keep, replace=False)

    # Combine all indices back together
    final_train_indices = np.concatenate([sampled_no_change_idx, idx_change_left, idx_change_right])

    # Create the new under-sampled training sets
    # Shuffle them so the model doesn't see all '0's then all '1's
    X_train_sub = X_train.loc[final_train_indices].sample(frac=1, random_state=42)
    y_train_sub = y_train.loc[X_train_sub.index]

    print(f"Original Training Size: {len(y_train)}")
    print(f"Under-sampled Training Size: {len(y_train_sub)}")
    print("New Class Distribution:\n", y_train_sub.value_counts())
    
    #X_train, X_test, y_train, y_test
    return X_train_sub, X_test, y_train_sub, y_test

def check_data_leakage(df, X_train, X_test):

    no_data_leakage = False
    # 1. Extract the unique Vehicle IDs from both sets
    # Note: Since X_train_sub and X_test are slices of the original df, 
    # we reference the original 'Veh_ID' column using their indices.

    train_ids = set(df.loc[X_train.index, 'Vehicle_Global_ID'].unique())
    test_ids = set(df.loc[X_test.index, 'Vehicle_Global_ID'].unique())

    # 2. Find the intersection
    overlapping_ids = train_ids.intersection(test_ids)

    # 3. Report the results
    print("--- Data Leakage Report ---")
    print(f"Unique Vehicles in Training: {len(train_ids)}")
    print(f"Unique Vehicles in Testing:  {len(test_ids)}")

    if len(overlapping_ids) == 0:
        print("SUCCESS: No data leakage detected. Training and Test sets are fully independent by Vehicle ID.")
        no_data_leakage = True
    else:
        no_data_leakage = False
        print(f"WARNING: Leakage detected! {len(overlapping_ids)} vehicles appear in both sets.")
        print(f"Overlapping IDs: {list(overlapping_ids)[:10]}...") 
        

    return no_data_leakage