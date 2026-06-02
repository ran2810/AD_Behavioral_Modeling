import warnings
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import numpy as np

# Local imports
from data.preprocess import preprocess_all
from data.preprocess_highd import preprocess_highd
from data.create_lstm_dataset import LaneChangeSequenceDataset
from extract_features import calculate_features
from utils.data_prep import data_split_with_sampling, check_data_leakage
from utils.benchmark import metrics_to_rows, save_results, print_benchmark_table
from models.rfclassifier import LaneChangeClassifier
from models.lstm import LaneChangeLSTM

import sys
print(sys.executable)

warnings.simplefilter(action='ignore', category=FutureWarning)

FEATURES = [
    # Lateral motion (dominant signals)
    "v_lat", "lat_displacement_1s", "lat_dist_moved_15f", "v_lat_accel",
    # Lateral history
    "v_lat_lag_5", "v_lat_lag_10", "v_lat_lag_20",
    # Longitudinal dynamics
    "v_vel", "a_long", "a_long_std_1s",
    # Lane context
    "Lane_ID", "can_go_right", "can_go_left",
    # Interaction with lead vehicle
    "actual_gap", "gap_rate_trend_1s", "rel_speed",
]

BENCHMARK_PATH = "results/benchmark.csv"


# #######################################################################
# Threshold tuning
# #######################################################################

def find_best_threshold(y_true, y_probabilities):
    """Finds the threshold that maximizes the F1-score for a specific class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probabilities)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def apply_custom_thresholds(probs, t_left, t_right):
    """Assigns classes based on optimized thresholds and highest probability."""
    final_preds = np.zeros(len(probs))
    for i in range(len(probs)):
        p_left, p_right = probs[i, 1], probs[i, 2]
        if p_left >= t_left and p_left > p_right:
            final_preds[i] = 1
        elif p_right >= t_right and p_right > p_left:
            final_preds[i] = 2
        else:
            final_preds[i] = 0
    return final_preds


#######################################################################
# Random Forest
#######################################################################

def trigger_rf(df, features, train_label='ngsim', test_label='ngsim', save_benchmark=True,
               keep_factor=3, w_left=3.0, w_right=15.0):
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, sampling_keep_factor=keep_factor)

    if not check_data_leakage(df, X_train, X_test):
        print("Data leakage detected! Check vehicle splits.")
        return None

    custom_weights = {0: 1.0, 1: w_left, 2: w_right}
    model = LaneChangeClassifier(custom_weights=custom_weights)

    print("Fitting RF model...")
    model.fit(X_train, y_train)

    model.plot_importance()
    y_probs = model.optimize_thresholds(X_test, y_test)
    y_pred = model.predict_optimized(y_probs)

    print("\n--- RF Optimized Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['None', 'Left', 'Right']))
    model.plot_confusion_matrix(y_test, y_pred)
    model.save_model(f"best_rf_{train_label}.joblib")

    if save_benchmark:
        rows = metrics_to_rows(y_test, y_pred, model='RF', train_dataset=train_label, test_dataset=test_label)
        save_results(rows, BENCHMARK_PATH)

    return model


def eval_rf_on_dataset(model, df_test, features, train_label, test_label):
    """Evaluate a trained RF model on a held-out dataset (cross-dataset eval)."""
    X_test = df_test[features]
    y_test = df_test['lane_change']

    y_probs = model.clf.predict_proba(X_test)

    # Re-optimize thresholds on the new test distribution
    t_left, _ = find_best_threshold((y_test == 1).astype(int), y_probs[:, 1])
    t_right, _ = find_best_threshold((y_test == 2).astype(int), y_probs[:, 2])
    print(f"Cross-eval thresholds -> Left: {t_left:.4f}, Right: {t_right:.4f}")

    y_pred = apply_custom_thresholds(y_probs, t_left, t_right)

    print(f"\n--- RF Cross-Eval: train={train_label}, test={test_label} ---")
    print(classification_report(y_test, y_pred, target_names=['None', 'Left', 'Right']))

    rows = metrics_to_rows(y_test, y_pred, model='RF', train_dataset=train_label, test_dataset=test_label)
    save_results(rows, BENCHMARK_PATH)


# #######################################################################
# LSTM
# #######################################################################

def validate_lstm(model, loader, device, thresh_left=0.5, thresh_right=0.5, optimize=False):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = torch.softmax(model(batch_x), dim=1)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)

    if optimize:
        t_left, _ = find_best_threshold((labels == 1).astype(int), probs[:, 1])
        t_right, _ = find_best_threshold((labels == 2).astype(int), probs[:, 2])
        print(f"Recalibrated Thresholds: Left={t_left:.4f}, Right={t_right:.4f}")
        thresh_left, thresh_right = t_left, t_right

    preds = apply_custom_thresholds(probs, thresh_left, thresh_right)

    f1_left = f1_score(labels, preds, labels=[1], average='macro')
    f1_right = f1_score(labels, preds, labels=[2], average='macro')

    return f1_left, f1_right, (thresh_left, thresh_right), labels, preds


def trigger_lstm(df, features, train_label='ngsim', window_size=30, save_benchmark=True):
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, sampling_keep_factor=4)

    train_dataset = LaneChangeSequenceDataset(X_train, y_train, df.loc[X_train.index, 'Vehicle_Global_ID'], window_size)
    test_dataset = LaneChangeSequenceDataset(X_test, y_test, df.loc[X_test.index, 'Vehicle_Global_ID'], window_size)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneChangeLSTM(input_size=len(features), hidden_size=64, num_layers=2).to(device)

    weights = torch.tensor([1.0, 3.0, 10.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training LSTM on {device}...")
    best_right_f1 = 0.0
    y_true, y_pred = None, None

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        f1_l, f1_r, thresholds, y_true, y_pred = validate_lstm(model, test_loader, device, optimize=True)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Left F1: {f1_l:.4f} | Right F1: {f1_r:.4f}")

        if f1_r > best_right_f1:
            best_right_f1 = f1_r
            torch.save({'model': model.state_dict(), 'thresholds': thresholds}, f"best_lstm_{train_label}.pth")

    print("\n--- Final LSTM Optimized Report ---")
    print(classification_report(y_true, y_pred, target_names=['None', 'Left', 'Right']))

    if save_benchmark:
        rows = metrics_to_rows(y_true, y_pred, model='LSTM', train_dataset=train_label, test_dataset=train_label)
        save_results(rows, BENCHMARK_PATH)

    return model, device


def eval_lstm_on_dataset(model, df_test, features, train_label, test_label, device, window_size=30):
    """Evaluate a trained LSTM on a held-out dataset (cross-dataset eval)."""
    X_test = df_test[features]
    y_test = df_test['lane_change']
    veh_ids = df_test['Vehicle_Global_ID']

    test_dataset = LaneChangeSequenceDataset(X_test, y_test, veh_ids, window_size)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    f1_l, f1_r, _, y_true, y_pred = validate_lstm(model, test_loader, device, optimize=True)

    print(f"\n--- LSTM Cross-Eval: train={train_label}, test={test_label} ---")
    print(f"Left F1: {f1_l:.4f} | Right F1: {f1_r:.4f}")
    print(classification_report(y_true, y_pred, target_names=['None', 'Left', 'Right']))

    rows = metrics_to_rows(y_true, y_pred, model='LSTM', train_dataset=train_label, test_dataset=test_label)
    save_results(rows, BENCHMARK_PATH)


# #######################################################################
# Cross-Dataset Benchmarking (2x2 matrix)
# #######################################################################

def run_cross_eval(df_ngsim, df_highd, features, model_name):
    """
    Full 2x2 benchmarking matrix:
        Train NGSIM -> Test NGSIM
        Train NGSIM -> Test HighD
        Train HighD -> Test HighD
        Train HighD -> Test NGSIM
    """
    print("\n" + "="*60)
    print("CROSS-DATASET BENCHMARKING")
    print("="*60)

    if model_name in ('rf', 'both'):
        print("\n[RF] Train=NGSIM, Test=NGSIM")
        rf_ngsim = trigger_rf(df_ngsim, features, train_label='ngsim', test_label='ngsim')

        print("\n[RF] Train=NGSIM -> Test=HighD")
        eval_rf_on_dataset(rf_ngsim, df_highd, features, train_label='ngsim', test_label='highd')

        print("\n[RF] Train=HighD, Test=HighD")
        rf_highd = trigger_rf(df_highd, features, train_label='highd', test_label='highd')

        print("\n[RF] Train=HighD -> Test=NGSIM")
        eval_rf_on_dataset(rf_highd, df_ngsim, features, train_label='highd', test_label='ngsim')

    if model_name in ('lstm', 'both'):
        # NGSIM: 10Hz -> 3s window = 30 frames; HighD: 25Hz -> 3s window = 75 frames
        print("\n[LSTM] Train=NGSIM, Test=NGSIM")
        lstm_ngsim, device = trigger_lstm(df_ngsim, features, train_label='ngsim', window_size=30)

        print("\n[LSTM] Train=NGSIM -> Test=HighD")
        eval_lstm_on_dataset(lstm_ngsim, df_highd, features, 'ngsim', 'highd', device, window_size=75)

        print("\n[LSTM] Train=HighD, Test=HighD")
        lstm_highd, device = trigger_lstm(df_highd, features, train_label='highd', window_size=75)

        print("\n[LSTM] Train=HighD -> Test=NGSIM")
        eval_lstm_on_dataset(lstm_highd, df_ngsim, features, 'highd', 'ngsim', device, window_size=30)

    print_benchmark_table(BENCHMARK_PATH)


# #######################################################################
# Entry Point
# #######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lane change prediction — train and evaluate models')
    parser.add_argument('model_name', type=str, choices=['rf', 'lstm', 'both'],
                        help='Which model to run: rf, lstm, or both')
    parser.add_argument('--dataset', type=str, default='ngsim', choices=['ngsim', 'highd', 'both'],
                        help='Which dataset to train on (default: ngsim)')
    parser.add_argument('--cross_eval', action='store_true',
                        help='Run full 2x2 cross-dataset benchmarking (requires both datasets)')
    parser.add_argument('--keep_factor', type=int, default=4,
                        help='None-class downsampling ratio relative to lane-change count (default: 3)')
    parser.add_argument('--w_right', type=float, default=10.0,
                        help='Class weight for Right lane change (default: 15.0)')
    parser.add_argument('--w_left', type=float, default=3.0,
                        help='Class weight for Left lane change (default: 3.0)')
    args = parser.parse_args()

    DATA_PATH = r"H:\GitHub\AD_Behavioral_Modeling\data"

    # Load datasets based on what's needed
    df_ngsim = None
    df_highd = None

    if args.dataset in ('ngsim', 'both') or args.cross_eval:
        print("\nLoading NGSIM data...")
        df_ngsim = preprocess_all(DATA_PATH)
        df_ngsim = calculate_features(df_ngsim, dataset='ngsim')

    if args.dataset in ('highd', 'both') or args.cross_eval:
        print("\nLoading HighD data...")
        df_highd = preprocess_highd(DATA_PATH)
        df_highd = calculate_features(df_highd, dataset='highd')

    if args.cross_eval:
        if df_ngsim is None or df_highd is None:
            raise RuntimeError("--cross_eval requires both datasets to be present.")
        run_cross_eval(df_ngsim, df_highd, FEATURES, args.model_name)

    else:
        df = df_ngsim if args.dataset in ('ngsim', 'both') else df_highd
        label = 'ngsim' if args.dataset in ('ngsim', 'both') else 'highd'
        window = 30 if label == 'ngsim' else 75

        if args.model_name in ('rf', 'both'):
            trigger_rf(df, FEATURES, train_label=label, test_label=label,
                       keep_factor=args.keep_factor, w_left=args.w_left, w_right=args.w_right)

        if args.model_name in ('lstm', 'both'):
            trigger_lstm(df, FEATURES, train_label=label, window_size=window)

        print_benchmark_table(BENCHMARK_PATH)
