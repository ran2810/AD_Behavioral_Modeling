import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def data_split_with_sampling(df, features, sampling_keep_factor=4):
    """
    data split into train and test by removing the boring data (driving straight without lane changes)
    """
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


def lstm_train_test_split(df, features, test_size=0.2, random_state=42):
    """
    Vehicle-grouped train/test split for LSTM.
    Returns full sorted trajectories — NO undersampling — so the dataset
    can build contiguous sliding windows. Class imbalance is handled by
    inverse-frequency WeightedRandomSampler inside LaneChangeSequenceDataset.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df[features], df['lane_change'], groups=df['Vehicle_Global_ID']))

    # Sort each split by vehicle + frame so windows are temporally contiguous
    df_train = df.iloc[train_idx].sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)
    df_test  = df.iloc[test_idx].sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)

    print(f"LSTM split — Train vehicles: {df_train['Vehicle_Global_ID'].nunique()}  "
          f"Test vehicles: {df_test['Vehicle_Global_ID'].nunique()}")
    print("Train labels:\n", df_train['lane_change'].value_counts().to_string())

    return df_train, df_test


def check_data_leakage(df, X_train, X_test):
    """
    check data leakage - same vehicle ID exists in train and test
    """

    no_data_leakage = False
    # Extract the unique Vehicle IDs from both sets
    # Note: Since X_train_sub and X_test are slices of the original df, 
    train_ids = set(df.loc[X_train.index, 'Vehicle_Global_ID'].unique())
    test_ids = set(df.loc[X_test.index, 'Vehicle_Global_ID'].unique())

    # Find the intersection
    overlapping_ids = train_ids.intersection(test_ids)

    # Report the results
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