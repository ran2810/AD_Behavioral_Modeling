import gc
import os
import warnings
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from tqdm import tqdm
import numpy as np

from data.preprocess import preprocess_all
from data.preprocess_highd import preprocess_highd
from data.create_lstm_dataset import LaneChangeSequenceDataset
from extract_features import calculate_features
from utils.data_prep import three_way_split, undersample_none
from utils.benchmark import metrics_to_rows, save_results, print_benchmark_table
from models.rfclassifier import LaneChangeClassifier
from models.lstm import LaneChangeLSTM

warnings.simplefilter(action='ignore', category=FutureWarning)

FEATURES = [
    "v_lat", "lat_displacement_1s", "lat_dist_moved_15f", "v_lat_accel",
    "v_lat_lag_5", "v_lat_lag_10", "v_lat_lag_20",
    "v_vel", "a_long", "a_long_std_1s",
    "Lane_ID", "can_go_right", "can_go_left",
    "actual_gap", "gap_rate_trend_1s", "rel_speed",
]

BENCHMARK_PATH = os.path.join("results", "benchmark.csv")

EXPERIMENTS = [
    'rf_ngsim', 'rf_highd', 'rf_ngsim_highd', 'rf_highd_ngsim',
    'lstm_ngsim', 'lstm_highd', 'lstm_ngsim_highd', 'lstm_highd_ngsim',
    'all'
]


#######################################################################
# Helpers
#######################################################################

def find_best_threshold(y_true, y_probabilities):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probabilities)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def apply_custom_thresholds(probs, t_left, t_right):
    final_preds = np.zeros(len(probs))
    # assign highest-confidence class that clears its threshold; default to None
    for i in range(len(probs)):
        p_left, p_right = probs[i, 1], probs[i, 2]
        if p_left >= t_left and p_left > p_right:
            final_preds[i] = 1
        elif p_right >= t_right and p_right > p_left:
            final_preds[i] = 2
        else:
            final_preds[i] = 0
    return final_preds


def _load_dataset(data_path, dataset):
    """Load + feature-engineer one dataset."""
    print(f"\nLoading {dataset.upper()}...")
    if dataset == 'ngsim':
        df = preprocess_all(data_path)
        df = calculate_features(df, dataset='ngsim')
    else:
        df = preprocess_highd(data_path)
        df = calculate_features(df, dataset='highd')
    gc.collect()
    return df


def _free_gpu():
    """Release fragmented CUDA memory between training runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


#######################################################################
# RF — core functions
#######################################################################

def _fit_rf(df_train, df_val, features, train_label, keep_factor, w_left, w_right):
    """
    Fit RF on df_train (undersampled), optimize thresholds on df_val.
    Thresholds stored in model — test set never touched here.
    """
    X_train, y_train = undersample_none(df_train[features], df_train['lane_change'], keep_factor)
    X_val,   y_val   = df_val[features], df_val['lane_change']

    model = LaneChangeClassifier(custom_weights={0: 1.0, 1: w_left, 2: w_right})
    print("Fitting RF...")
    model.fit(X_train, y_train)
    del X_train, y_train; gc.collect()

    # threshold tuned on val — not on test
    model.optimize_thresholds(X_val, y_val)
    del X_val, y_val; gc.collect()

    results_dir = os.path.join("results", train_label)
    model.plot_importance(save_path=os.path.join(results_dir, "rf_feature_importance.png"))
    model.save_model(os.path.join("models", f"best_rf_{train_label}.joblib"))

    return model


def _eval_rf(model, df_test, features, train_label, test_label):
    """Evaluate RF with val-optimized thresholds on df_test. Saves CM + benchmark row."""
    X_test = df_test[features]
    y_test = df_test['lane_change']

    probs  = model.clf.predict_proba(X_test)
    # val thresholds already stored in model.thresh_left / thresh_right
    y_pred = model.predict_optimized(probs)

    tag = f"{train_label}_to_{test_label}"
    print(f"\n--- RF | train={train_label} test={test_label} ---")
    print(f"Thresholds  Left={model.thresh_left:.4f}  Right={model.thresh_right:.4f}")
    print(classification_report(y_test, y_pred, target_names=['None', 'Left', 'Right']))

    results_dir = os.path.join("results", tag)
    model.plot_confusion_matrix(y_test, y_pred,
                                save_path=os.path.join(results_dir, "rf_confusion_matrix.png"))

    rows = metrics_to_rows(y_test, y_pred, model='RF',
                           train_dataset=train_label, test_dataset=test_label)
    save_results(rows, BENCHMARK_PATH)

    del X_test, probs, y_pred; gc.collect()


#######################################################################
# LSTM — core functions
#######################################################################

def validate_lstm(model, loader, device, thresh_left=0.5, thresh_right=0.5, optimize=False):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        # collect softmax probabilities across all batches
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = torch.softmax(model(batch_x), dim=1)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)

    if optimize:
        # find per-class thresholds that maximize F1 on this set (val, not test)
        t_left, _  = find_best_threshold((labels == 1).astype(int), probs[:, 1])
        t_right, _ = find_best_threshold((labels == 2).astype(int), probs[:, 2])
        thresh_left, thresh_right = t_left, t_right

    preds    = apply_custom_thresholds(probs, thresh_left, thresh_right)
    f1_left  = f1_score(labels, preds, labels=[1], average='macro')
    f1_right = f1_score(labels, preds, labels=[2], average='macro')

    return f1_left, f1_right, (thresh_left, thresh_right), labels, preds


def permutation_importance_lstm(model, dataset, features, device, n_repeats=2):
    """
    Permutation importance: zero out the predictive power of one feature at a time
    by shuffling it across all sequences, then measure the drop in macro F1.
    n_repeats averages out the randomness of the shuffle.
    """
    loader   = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    f1_l, f1_r, _, _, _ = validate_lstm(model, loader, device)
    baseline = (f1_l + f1_r) / 2

    importances = {}
    X_backup = dataset.X_values.copy()  # save original to restore after each feature

    for fi, feat_name in enumerate(features):
        drops = []
        # repeat the shuffle to reduce variance
        for _ in range(n_repeats):
            # shuffle this feature across all rows — breaks its correlation with the label
            dataset.X_values[:, fi] = np.random.permutation(dataset.X_values[:, fi])
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
            f1_l, f1_r, _, _, _ = validate_lstm(model, loader, device)
            drops.append(baseline - (f1_l + f1_r) / 2)

        importances[feat_name] = round(float(np.mean(drops)), 4)
        # restore the column before moving to the next feature
        dataset.X_values[:, fi] = X_backup[:, fi]

    del X_backup; gc.collect()
    return importances


def _make_lstm_dataset(df, features, window_size):
    """Sort by vehicle+frame so sliding windows are temporally contiguous."""
    df_s = df.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID']).reset_index(drop=True)
    return LaneChangeSequenceDataset(df_s[features], df_s['lane_change'],
                                     df_s['Vehicle_Global_ID'], window_size)


def _train_lstm(train_dataset, val_dataset, features, train_label,
                window_size, device, n_epochs=10):
    """
    Training loop with val-based model selection.
    Thresholds are optimized on val each epoch — best checkpoint reloaded after training.
    Test set is never seen here.
    """
    # cap epoch to 6x minority sequences — keeps per-epoch time manageable on CPU/GPU
    labels_arr = np.array(train_dataset.labels)
    n_minority = int((labels_arr != 0).sum())
    epoch_size = min(len(train_dataset), n_minority * 6)
    sampler    = WeightedRandomSampler(train_dataset.sample_weights,
                                       num_samples=epoch_size, replacement=True)
    print(f"Epoch size: {epoch_size:,} ({epoch_size // 64:,} batches/epoch)")

    # num_workers=0 + pin_memory=False — avoids forking RAM copies on Colab
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = LaneChangeLSTM(input_size=len(features), hidden_size=64, num_layers=2).to(device)
    # class weights: Right (2) gets 10x to compensate for ~3x fewer samples than Left
    weights   = torch.tensor([1.0, 3.0, 10.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_ckpt       = os.path.join("models", f"best_lstm_{train_label}.pth")
    best_right_f1   = 0.0
    best_thresholds = (0.5, 0.5)
    os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)

    print(f"Training LSTM on {device}...")
    epoch_bar = tqdm(range(n_epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0

        # --- inner batch loop ---
        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}",
                         unit="batch", leave=False, dynamic_ncols=True)
        for batch_x, batch_y in batch_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)  # set_to_none frees grad tensors immediately
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- val evaluation after each epoch ---
        # thresholds optimized on val — Right F1 drives model selection
        f1_l, f1_r, thresholds, _, _ = validate_lstm(model, val_loader, device, optimize=True)
        avg_loss = total_loss / len(train_loader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_l=f"{f1_l:.3f}", val_r=f"{f1_r:.3f}")
        tqdm.write(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Val Left: {f1_l:.4f} | Val Right: {f1_r:.4f}")

        # save only when val Right F1 improves — avoids saving on every epoch
        if f1_r > best_right_f1:
            best_right_f1   = f1_r
            best_thresholds = thresholds
            model.save_model(best_ckpt, features, thresholds)

    # reload best checkpoint so final eval uses the best-performing state
    if os.path.exists(best_ckpt):
        saved_model, _, best_thresholds = LaneChangeLSTM.load_model(best_ckpt)
        model.load_state_dict(saved_model.state_dict())
        model = model.to(device)

    del train_loader, train_dataset, labels_arr
    _free_gpu(); gc.collect()

    return model, best_thresholds


def _eval_lstm(model, test_dataset, features, train_label, test_label, device, thresholds):
    """
    Final evaluation on test set.
    Uses val-optimized thresholds — no re-fitting on test.
    """
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=False)

    # optimize=False — use the thresholds we found on val, not on this test set
    f1_l, f1_r, _, y_true, y_pred = validate_lstm(
        model, test_loader, device,
        thresh_left=thresholds[0], thresh_right=thresholds[1],
        optimize=False
    )

    tag = f"{train_label}_to_{test_label}"
    print(f"\n--- LSTM | train={train_label} test={test_label} ---")
    print(f"Thresholds  Left={thresholds[0]:.4f}  Right={thresholds[1]:.4f}")
    print(f"Left F1: {f1_l:.4f} | Right F1: {f1_r:.4f}")
    print(classification_report(y_true, y_pred, target_names=['None', 'Left', 'Right']))

    results_dir = os.path.join("results", tag)
    model.plot_confusion_matrix(y_true, y_pred,
                                save_path=os.path.join(results_dir, "lstm_confusion_matrix.png"))

    rows = metrics_to_rows(y_true, y_pred, model='LSTM',
                           train_dataset=train_label, test_dataset=test_label)
    save_results(rows, BENCHMARK_PATH)

    del test_loader; gc.collect()
    return y_true, y_pred


#######################################################################
# Experiment runners — one per row in the table
#######################################################################

def run_rf_indomain(data_path, features, label, keep_factor, w_left, w_right):
    """RF In-domain: train/val/test all from same dataset."""
    df = _load_dataset(data_path, label)
    df_train, df_val, df_test = three_way_split(df, features)
    del df; gc.collect()

    model = _fit_rf(df_train, df_val, features, label, keep_factor, w_left, w_right)
    del df_train, df_val; gc.collect()

    _eval_rf(model, df_test, features, label, label)
    del df_test; gc.collect()


def rf_ngsim_highd(data_path, features, keep_factor, w_left, w_right):
    """
    RF Cross-domain: train on NGSIM, val on NGSIM, test on HighD.
    Staged — NGSIM freed before HighD is loaded.
    """
    # load NGSIM, extract train/val, copy before freeing parent df
    df_ngsim = _load_dataset(data_path, 'ngsim')
    df_train, df_val, _ = three_way_split(df_ngsim, features)
    df_train = df_train.copy(); df_val = df_val.copy()
    del df_ngsim; gc.collect()

    model = _fit_rf(df_train, df_val, features, 'ngsim_cross', keep_factor, w_left, w_right)
    del df_train, df_val; gc.collect()

    # load HighD, extract only test split, free HighD df
    df_highd = _load_dataset(data_path, 'highd')
    _, _, df_highd_test = three_way_split(df_highd, features)
    df_highd_test = df_highd_test.copy()  # own copy before parent is freed
    del df_highd; gc.collect()

    _eval_rf(model, df_highd_test, features, 'ngsim', 'highd')
    del df_highd_test; gc.collect()


def run_lstm_indomain(data_path, features, label, window_size):
    """LSTM In-domain: train/val/test all from same dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = _load_dataset(data_path, label)
    df_train, df_val, df_test = three_way_split(df, features)
    del df; gc.collect()

    # build datasets one at a time, freeing the raw df slice after each
    train_ds = _make_lstm_dataset(df_train, features, window_size)
    del df_train; gc.collect()
    val_ds   = _make_lstm_dataset(df_val,   features, window_size)
    del df_val; gc.collect()
    test_ds  = _make_lstm_dataset(df_test,  features, window_size)
    del df_test; gc.collect()

    model, thresholds = _train_lstm(train_ds, val_ds, features, label, window_size, device)
    del val_ds; gc.collect()

    print("Computing LSTM feature importance (permutation)...")
    importances = permutation_importance_lstm(model, test_ds, features, device)
    model.plot_feature_importance(importances, features,
                                  save_path=os.path.join("results", label, "lstm_feature_importance.png"))

    _eval_lstm(model, test_ds, features, label, label, device, thresholds)
    del test_ds; _free_gpu(); gc.collect()


def lstm_ngsim_highd(data_path, features, window_size_src=30, window_size_tgt=75):
    """
    LSTM Cross-domain: train on NGSIM, val on NGSIM, test on HighD.
    Staged — NGSIM freed before HighD is loaded.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load NGSIM, build train/val sequence datasets, free df
    df_ngsim = _load_dataset(data_path, 'ngsim')
    df_train, df_val, _ = three_way_split(df_ngsim, features)
    df_train = df_train.copy(); df_val = df_val.copy()
    del df_ngsim; gc.collect()

    train_ds = _make_lstm_dataset(df_train, features, window_size_src)
    del df_train; gc.collect()
    val_ds   = _make_lstm_dataset(df_val,   features, window_size_src)
    del df_val; gc.collect()

    # train on NGSIM — NGSIM df fully freed before HighD is loaded
    model, thresholds = _train_lstm(train_ds, val_ds, features, 'ngsim_cross',
                                    window_size_src, device)
    del train_ds, val_ds; _free_gpu(); gc.collect()

    # load HighD, extract test split only, free HighD df
    df_highd = _load_dataset(data_path, 'highd')
    _, _, df_highd_test = three_way_split(df_highd, features)
    df_highd_test = df_highd_test.copy()
    del df_highd; gc.collect()

    test_ds = _make_lstm_dataset(df_highd_test, features, window_size_tgt)
    del df_highd_test; gc.collect()

    _eval_lstm(model, test_ds, features, 'ngsim', 'highd', device, thresholds)
    del test_ds; _free_gpu(); gc.collect()


def rf_highd_ngsim(data_path, features, keep_factor, w_left, w_right):
    """
    RF Cross-domain (reverse): train on HighD, val on HighD, test on NGSIM.
    Staged — HighD freed before NGSIM is loaded.
    """
    # load HighD, extract train/val, copy before freeing parent df
    df_highd = _load_dataset(data_path, 'highd')
    df_train, df_val, _ = three_way_split(df_highd, features)
    df_train = df_train.copy(); df_val = df_val.copy()
    del df_highd; gc.collect()

    model = _fit_rf(df_train, df_val, features, 'highd_cross', keep_factor, w_left, w_right)
    del df_train, df_val; gc.collect()

    # load NGSIM, extract only test split, free NGSIM df
    df_ngsim = _load_dataset(data_path, 'ngsim')
    _, _, df_ngsim_test = three_way_split(df_ngsim, features)
    df_ngsim_test = df_ngsim_test.copy()  # own copy before parent is freed
    del df_ngsim; gc.collect()

    _eval_rf(model, df_ngsim_test, features, 'highd', 'ngsim')
    del df_ngsim_test; gc.collect()


def lstm_highd_ngsim(data_path, features, window_size_src=75, window_size_tgt=30):
    """
    LSTM Cross-domain (reverse): train on HighD, val on HighD, test on NGSIM.
    Staged — HighD freed before NGSIM is loaded.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load HighD, build train/val sequence datasets, free df
    df_highd = _load_dataset(data_path, 'highd')
    df_train, df_val, _ = three_way_split(df_highd, features)
    df_train = df_train.copy(); df_val = df_val.copy()
    del df_highd; gc.collect()

    train_ds = _make_lstm_dataset(df_train, features, window_size_src)
    del df_train; gc.collect()
    val_ds   = _make_lstm_dataset(df_val,   features, window_size_src)
    del df_val; gc.collect()

    # train on HighD — HighD df fully freed before NGSIM is loaded
    model, thresholds = _train_lstm(train_ds, val_ds, features, 'highd_cross',
                                    window_size_src, device)
    del train_ds, val_ds; _free_gpu(); gc.collect()

    # load NGSIM, extract test split only, free NGSIM df
    df_ngsim = _load_dataset(data_path, 'ngsim')
    _, _, df_ngsim_test = three_way_split(df_ngsim, features)
    df_ngsim_test = df_ngsim_test.copy()
    del df_ngsim; gc.collect()

    test_ds = _make_lstm_dataset(df_ngsim_test, features, window_size_tgt)
    del df_ngsim_test; gc.collect()

    _eval_lstm(model, test_ds, features, 'highd', 'ngsim', device, thresholds)
    del test_ds; _free_gpu(); gc.collect()


#######################################################################
# All-experiment runner — staged loading, one dataset in RAM at a time
#######################################################################

def run_all(data_path, features, keep_factor, w_left, w_right):
    print("\n" + "="*60)
    print("RUNNING ALL 8 EXPERIMENTS")
    print("="*60)

    # --- NGSIM in-domain: load once, run RF + LSTM, keep train/val for cross ---
    print("\n--- NGSIM In-domain ---")
    df = _load_dataset(data_path, 'ngsim')
    df_train, df_val, df_test = three_way_split(df, features)
    del df; gc.collect()

    print("\n[1/8] RF In-domain NGSIM")
    model_rf_n = _fit_rf(df_train, df_val, features, 'ngsim', keep_factor, w_left, w_right)
    _eval_rf(model_rf_n, df_test, features, 'ngsim', 'ngsim')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n[4/8] LSTM In-domain NGSIM")
    # build sequence datasets — free each df slice immediately after
    train_ds = _make_lstm_dataset(df_train, features, 30)
    val_ds   = _make_lstm_dataset(df_val,   features, 30)
    test_ds  = _make_lstm_dataset(df_test,  features, 30)
    model_lstm_n, thresh_n = _train_lstm(train_ds, val_ds, features, 'ngsim', 30, device)
    del train_ds, val_ds; gc.collect()
    importances = permutation_importance_lstm(model_lstm_n, test_ds, features, device)
    model_lstm_n.plot_feature_importance(importances, features,
                                         save_path=os.path.join("results", "ngsim", "lstm_feature_importance.png"))
    _eval_lstm(model_lstm_n, test_ds, features, 'ngsim', 'ngsim', device, thresh_n)
    del test_ds; _free_gpu()

    # save copies of NGSIM train/val/test — train+val for NGSIM→HighD cross, test for HighD→NGSIM cross
    df_ngsim_train = df_train.copy()
    df_ngsim_val   = df_val.copy()
    df_ngsim_test  = df_test.copy()
    del df_train, df_val, df_test; gc.collect()

    # --- HighD in-domain: load once, run RF + LSTM ---
    print("\n--- HighD In-domain ---")
    df = _load_dataset(data_path, 'highd')
    df_train, df_val, df_test = three_way_split(df, features)
    del df; gc.collect()

    print("\n[2/8] RF In-domain HighD")
    model_rf_h = _fit_rf(df_train, df_val, features, 'highd', keep_factor, w_left, w_right)
    _eval_rf(model_rf_h, df_test, features, 'highd', 'highd')

    print("\n[5/8] LSTM In-domain HighD")
    train_ds = _make_lstm_dataset(df_train, features, 75)
    val_ds   = _make_lstm_dataset(df_val,   features, 75)
    test_ds  = _make_lstm_dataset(df_test,  features, 75)
    model_lstm_h, thresh_h = _train_lstm(train_ds, val_ds, features, 'highd', 75, device)
    del train_ds, val_ds; gc.collect()
    importances = permutation_importance_lstm(model_lstm_h, test_ds, features, device)
    model_lstm_h.plot_feature_importance(importances, features,
                                         save_path=os.path.join("results", "highd", "lstm_feature_importance.png"))
    _eval_lstm(model_lstm_h, test_ds, features, 'highd', 'highd', device, thresh_h)
    del test_ds; _free_gpu()

    # save HighD train/val/test — train+val for HighD→NGSIM cross, test for NGSIM→HighD cross
    # same random_state=42 seed guarantees these are the identical splits used in in-domain experiments
    df_highd_train = df_train.copy()
    df_highd_val   = df_val.copy()
    df_highd_test  = df_test.copy()
    del df_train, df_val, df_test; gc.collect()

    # --- Cross-domain: NGSIM train/val (already in memory) → HighD test ---
    print("\n--- Cross-domain ---")

    print("\n[3/8] RF Cross-domain (NGSIM → HighD)")
    # NGSIM train/val still in memory — no reload needed
    model_rf_cross = _fit_rf(df_ngsim_train, df_ngsim_val, features,
                              'ngsim_cross', keep_factor, w_left, w_right)
    _eval_rf(model_rf_cross, df_highd_test, features, 'ngsim', 'highd')

    print("\n[6/8] LSTM Cross-domain (NGSIM -> HighD)")
    # build NGSIM sequence datasets from the cached train/val splits
    train_ds = _make_lstm_dataset(df_ngsim_train, features, 30)
    val_ds   = _make_lstm_dataset(df_ngsim_val,   features, 30)
    del df_ngsim_train, df_ngsim_val; gc.collect()
    model_lstm_cross, thresh_cross = _train_lstm(train_ds, val_ds, features,
                                                  'ngsim_cross', 30, device)
    del train_ds, val_ds; _free_gpu(); gc.collect()
    # build HighD test dataset from the cached test split
    test_ds = _make_lstm_dataset(df_highd_test, features, 75)
    # df_highd_test still needed for experiment 7 -- keep it alive
    _eval_lstm(model_lstm_cross, test_ds, features, 'ngsim', 'highd', device, thresh_cross)
    del test_ds; _free_gpu(); gc.collect()

    # --- Reverse cross-domain: HighD train/val -> NGSIM test (no extra loads needed) ---
    print("\n[7/8] RF Cross-domain (HighD -> NGSIM)")
    # HighD train/val and NGSIM test are all cached from the in-domain passes
    model_rf_cross_rev = _fit_rf(df_highd_train, df_highd_val, features,
                                  'highd_cross', keep_factor, w_left, w_right)
    _eval_rf(model_rf_cross_rev, df_ngsim_test, features, 'highd', 'ngsim')
    del df_highd_test; gc.collect()

    print("\n[8/8] LSTM Cross-domain (HighD -> NGSIM)")
    # build HighD sequence datasets from the cached train/val splits
    train_ds = _make_lstm_dataset(df_highd_train, features, 75)
    val_ds   = _make_lstm_dataset(df_highd_val,   features, 75)
    del df_highd_train, df_highd_val; gc.collect()
    model_lstm_cross_rev, thresh_cross_rev = _train_lstm(train_ds, val_ds, features,
                                                          'highd_cross', 75, device)
    del train_ds, val_ds; _free_gpu(); gc.collect()
    # build NGSIM test dataset from the cached test split
    test_ds = _make_lstm_dataset(df_ngsim_test, features, 30)
    del df_ngsim_test; gc.collect()
    _eval_lstm(model_lstm_cross_rev, test_ds, features, 'highd', 'ngsim', device, thresh_cross_rev)
    del test_ds; _free_gpu(); gc.collect()

    print_benchmark_table(BENCHMARK_PATH)


#######################################################################
# Entry Point
#######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lane change prediction')
    parser.add_argument('--experiment', type=str, default='rf_ngsim', choices=EXPERIMENTS,
                        help='Which experiment to run (default: rf_ngsim)')
    parser.add_argument('--data_path',  type=str, default='data',
                        help='Path to data folder (default: data/)')
    parser.add_argument('--keep_factor', type=int,   default=3,
                        help='RF None-class downsampling factor (default: 3)')
    parser.add_argument('--w_right',     type=float, default=15.0,
                        help='RF class weight for Right (default: 15.0)')
    parser.add_argument('--w_left',      type=float, default=3.0,
                        help='RF class weight for Left (default: 3.0)')
    args = parser.parse_args()

    kw = dict(keep_factor=args.keep_factor, w_left=args.w_left, w_right=args.w_right)

    if args.experiment == 'rf_ngsim':
        # Train, Val and Test with NGSIM with ratios of 70, 15, 15
        run_rf_indomain(args.data_path, FEATURES, 'ngsim', **kw)

    elif args.experiment == 'rf_highd':
        # Train, Val and Test with HighD with ratios of 70, 15, 15
        run_rf_indomain(args.data_path, FEATURES, 'highd', **kw)

    elif args.experiment == 'rf_ngsim_highd':
        # NGSIM Train, NGSIM Val, HighD Test
        rf_ngsim_highd(args.data_path, FEATURES, **kw)

    elif args.experiment == 'rf_highd_ngsim':
        # HighD Train, HighD Val, NGSIM Test
        rf_highd_ngsim(args.data_path, FEATURES, **kw)

    elif args.experiment == 'lstm_ngsim':
        # Train, Val and Test with NGSIM with ratios of 70, 15, 15
        run_lstm_indomain(args.data_path, FEATURES, 'ngsim', window_size=30)

    elif args.experiment == 'lstm_highd':
        # Train, Val and Test with HighD with ratios of 70, 15, 15
        run_lstm_indomain(args.data_path, FEATURES, 'highd', window_size=75)

    elif args.experiment == 'lstm_ngsim_highd':
        # NGSIM Train, NGSIM Val, HighD Test
        lstm_ngsim_highd(args.data_path, FEATURES, window_size_src=30, window_size_tgt=75)

    elif args.experiment == 'lstm_highd_ngsim':
        # HighD Train, HighD Val, NGSIM Test
        lstm_highd_ngsim(args.data_path, FEATURES, window_size_src=75, window_size_tgt=30)

    elif args.experiment == 'all':
        run_all(args.data_path, FEATURES, **kw)

    print_benchmark_table(BENCHMARK_PATH)
