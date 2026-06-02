import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def three_way_split(df, features, random_state=42):
    """
    Vehicle-grouped 70 / 15 / 15 train-val-test split.
    No undersampling — returns raw dataframe slices.
    RF caller: apply undersample_none on the train slice.
    LSTM caller: sort by vehicle+frame before building sequences.
    """
    # 70% train, 30% temp
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss.split(df[features], df['lane_change'],
                                         groups=df['Vehicle_Global_ID']))

    df_train = df.iloc[train_idx]
    df_temp  = df.iloc[temp_idx]

    # split the 30% into val (15%) and test (15%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_idx, test_idx = next(gss2.split(df_temp[features], df_temp['lane_change'],
                                        groups=df_temp['Vehicle_Global_ID']))

    df_val  = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]

    print(f"Split  Train: {df_train['Vehicle_Global_ID'].nunique():,} vehicles "
          f"({len(df_train):,} rows) | "
          f"Val: {df_val['Vehicle_Global_ID'].nunique():,} | "
          f"Test: {df_test['Vehicle_Global_ID'].nunique():,}")

    return df_train, df_val, df_test


def undersample_none(X_train, y_train, keep_factor, random_state=42):
    """
    Undersample majority class (No Change = 0) in the training set.
    keep_factor: how many None rows to keep per lane-change row.
    """
    idx_none  = y_train[y_train == 0].index
    idx_left  = y_train[y_train == 1].index
    idx_right = y_train[y_train == 2].index

    n_changes = len(idx_left) + len(idx_right)
    n_keep    = min(len(idx_none), n_changes * keep_factor)
    sampled   = np.random.choice(idx_none, size=n_keep, replace=False)

    final_idx = np.concatenate([sampled, idx_left, idx_right])

    # shuffle so model doesn't see all 0s first
    X_sub = X_train.loc[final_idx].sample(frac=1, random_state=random_state)
    y_sub = y_train.loc[X_sub.index]

    print(f"Under-sampled train: {len(y_sub):,} | {y_sub.value_counts().to_dict()}")
    return X_sub, y_sub


# kept for backward compat — still used by tune_rf.py
def data_split_with_sampling(df, features, sampling_keep_factor=4):
    """80/20 vehicle-grouped split with None-class undersampling in train."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df[features], df['lane_change'],
                                         groups=df['Vehicle_Global_ID']))

    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx]['lane_change']
    X_test  = df.iloc[test_idx][features]
    y_test  = df.iloc[test_idx]['lane_change']

    idx_none  = y_train[y_train == 0].index
    idx_left  = y_train[y_train == 1].index
    idx_right = y_train[y_train == 2].index

    n_changes = len(idx_left) + len(idx_right)
    n_keep    = min(len(idx_none), n_changes * sampling_keep_factor)
    sampled   = np.random.choice(idx_none, size=n_keep, replace=False)

    final_idx = np.concatenate([sampled, idx_left, idx_right])
    X_train_sub = X_train.loc[final_idx].sample(frac=1, random_state=42)
    y_train_sub = y_train.loc[X_train_sub.index]

    print(f"Original Training Size: {len(y_train)}")
    print(f"Under-sampled Training Size: {len(y_train_sub)}")
    print("New Class Distribution:\n", y_train_sub.value_counts())

    return X_train_sub, X_test, y_train_sub, y_test


def check_data_leakage(df, X_train, X_test):
    """Check no vehicle ID appears in both train and test."""
    train_ids = set(df.loc[X_train.index, 'Vehicle_Global_ID'].unique())
    test_ids  = set(df.loc[X_test.index,  'Vehicle_Global_ID'].unique())
    overlap   = train_ids.intersection(test_ids)

    print("--- Data Leakage Report ---")
    print(f"Train vehicles: {len(train_ids)} | Test vehicles: {len(test_ids)}")

    if len(overlap) == 0:
        print("SUCCESS: No leakage — splits are fully independent by Vehicle ID.")
        return True

    print(f"WARNING: {len(overlap)} vehicles overlap! First 10: {list(overlap)[:10]}")
    return False


def lstm_train_test_split(df, features, test_size=0.2, random_state=42):
    """Legacy 80/20 LSTM split — kept for standalone lstm runs."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df[features], df['lane_change'],
                                         groups=df['Vehicle_Global_ID']))

    df_train = df.iloc[train_idx].sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)
    df_test  = df.iloc[test_idx].sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)

    print(f"LSTM split — Train: {df_train['Vehicle_Global_ID'].nunique()} vehicles | "
          f"Test: {df_test['Vehicle_Global_ID'].nunique()}")
    return df_train, df_test
