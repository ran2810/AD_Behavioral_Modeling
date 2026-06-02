"""
RF Hyperparameter Tuning
Grid search over keep_factor and w_right using a vehicle-stratified sample.
Saves results to results/tune_rf.csv.

Usage:
    python tune_rf.py                    # 30% vehicle sample (fast)
    python tune_rf.py --sample 1.0       # full data
"""
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
import os, sys

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from data.preprocess import preprocess_all
from extract_features import calculate_features


DATA_PATH = r"H:\GitHub\AD_Behavioral_Modeling\data"
OUT_PATH  = "results/tune_rf.csv"


def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[min(idx, len(thresholds) - 1)]


def apply_thresholds(probs, t_left, t_right):
    preds = np.zeros(len(probs))
    for i in range(len(probs)):
        pl, pr = probs[i, 1], probs[i, 2]
        if pl >= t_left and pl > pr:
            preds[i] = 1
        elif pr >= t_right and pr > pl:
            preds[i] = 2
    return preds


def sample_vehicles(df, frac=0.30, seed=42):
    """Keep a stratified random sample of vehicles (preserving full trajectories)."""
    vehicles = df['Vehicle_Global_ID'].unique()
    n = max(1, int(len(vehicles) * frac))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(vehicles, size=n, replace=False)
    return df[df['Vehicle_Global_ID'].isin(sampled)].reset_index(drop=True)


def run_trial(df, features, keep_factor, w_right, w_left=3.0, seed=42):
    """Single train/test split + RF fit + threshold-optimized eval."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df[features], df['lane_change'], groups=df['Vehicle_Global_ID']))

    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx]['lane_change']
    X_test  = df.iloc[test_idx][features]
    y_test  = df.iloc[test_idx]['lane_change']

    # Undersample None class
    idx_none  = y_train[y_train == 0].index
    idx_left  = y_train[y_train == 1].index
    idx_right = y_train[y_train == 2].index
    n_keep = min(len(idx_none), (len(idx_left) + len(idx_right)) * keep_factor)
    rng = np.random.default_rng(seed)
    sampled_none = rng.choice(idx_none, size=n_keep, replace=False)
    final_idx = np.concatenate([sampled_none, idx_left, idx_right])
    X_tr = X_train.loc[final_idx]
    y_tr = y_train.loc[final_idx]

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight={0: 1.0, 1: w_left, 2: w_right},
        max_depth=15,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=seed
    )
    clf.fit(X_tr, y_tr)

    probs = clf.predict_proba(X_test)
    t_left  = find_best_threshold((y_test == 1).astype(int), probs[:, 1])
    t_right = find_best_threshold((y_test == 2).astype(int), probs[:, 2])
    preds = apply_thresholds(probs, t_left, t_right)

    f1_none  = f1_score(y_test, preds, labels=[0], average='macro', zero_division=0)
    f1_left  = f1_score(y_test, preds, labels=[1], average='macro', zero_division=0)
    f1_right = f1_score(y_test, preds, labels=[2], average='macro', zero_division=0)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)

    return {
        'keep_factor': keep_factor,
        'w_right':     w_right,
        'w_left':      w_left,
        't_left':      round(t_left, 4),
        't_right':     round(t_right, 4),
        'f1_none':     round(f1_none, 4),
        'f1_left':     round(f1_left, 4),
        'f1_right':    round(f1_right, 4),
        'f1_macro':    round(f1_macro, 4),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=float, default=0.30,
                        help='Fraction of vehicles to use (default 0.30 for speed)')
    args = parser.parse_args()

    print("Loading NGSIM data...")
    df = preprocess_all(DATA_PATH)
    df = calculate_features(df, dataset='ngsim')

    if args.sample < 1.0:
        df = sample_vehicles(df, frac=args.sample)
        print(f"Using {args.sample*100:.0f}% vehicle sample: {len(df):,} rows, "
              f"{df['Vehicle_Global_ID'].nunique()} vehicles")

    features = [
        "v_lat", "lat_displacement_1s", "lat_dist_moved_15f", "v_lat_accel",
        "v_lat_lag_5", "v_lat_lag_10", "v_lat_lag_20",
        "v_vel", "a_long", "a_long_std_1s",
        "Lane_ID", "can_go_right", "can_go_left",
        "actual_gap", "gap_rate_trend_1s", "rel_speed",
    ]

    # Grid: keep_factor x w_right
    keep_factors = [1, 2, 3, 4]
    w_rights     = [10, 15, 20, 25, 30]

    total = len(keep_factors) * len(w_rights)
    results = []

    for i, kf in enumerate(keep_factors):
        for j, wr in enumerate(w_rights):
            trial_num = i * len(w_rights) + j + 1
            print(f"[{trial_num}/{total}] keep_factor={kf}  w_right={wr} ...", flush=True)
            row = run_trial(df, features, keep_factor=kf, w_right=wr)
            results.append(row)
            print(f"        -> F1 none={row['f1_none']} left={row['f1_left']} "
                  f"right={row['f1_right']}  macro={row['f1_macro']}", flush=True)

    res_df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
    os.makedirs('results', exist_ok=True)
    res_df.to_csv(OUT_PATH, index=False)


    print("GRID SEARCH RESULTS (sorted by macro F1)")
    print(res_df.to_string(index=False))
    print(f"\nBest: keep_factor={res_df.iloc[0]['keep_factor']}  "
          f"w_right={res_df.iloc[0]['w_right']}  "
          f"macro_f1={res_df.iloc[0]['f1_macro']}")
    print(f"Results saved to {OUT_PATH}")
