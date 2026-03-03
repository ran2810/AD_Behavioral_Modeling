import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_recall_curve)

# Local imports
from data.preprocess import preprocess_all
from extract_features import calculate_features
from utils.data_prep import data_split_with_sampling, check_data_leakage

import warnings

# Suppress all FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Helper Functions ---

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

def find_best_threshold(y_true, y_probabilities):
    """Finds the threshold that maximizes the F1-score for a specific class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probabilities)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds)-1)], f1_scores[best_idx]

def get_avg_prob_curve(test_results, class_idx, prob_col, window_size=30):
    """Calculates the mean probability leading up to the start of a lane change."""
    all_curves = []
    sorted_test = test_results.sort_values(['Vehicle_Global_ID', 'Frame_Global_ID'])
    
    # Detect 0 -> Class transition
    lc_starts = sorted_test[(sorted_test['actual_class'] == class_idx) & 
                            (sorted_test['actual_class'].shift(1) == 0) &
                            (sorted_test['Vehicle_Global_ID'] == sorted_test['Vehicle_Global_ID'].shift(1))]
    
    for idx in lc_starts.index:
        pos = sorted_test.index.get_loc(idx)
        if pos >= window_size:
            window = sorted_test.iloc[pos-window_size : pos+1]
            if window['Vehicle_Global_ID'].nunique() == 1:
                all_curves.append(window[prob_col].values)
                
    return np.mean(all_curves, axis=0) if all_curves else None

# --- Main Execution ---

def main():
    # 1. Data Loading and Feature Engineering
    DATA_PATH = r"H:\GitHub\AD_Behavioral_Modeling\data"
    df = preprocess_all(DATA_PATH)
    df = calculate_features(df)

    features = [
        "v_vel", "a_long", "v_lat", "Lane_ID", "v_lat_lag_5", "v_lat_lag_10", # "v_lat_accel", "lat_dist_moved_15f",  "a_lat",
         #"lat_displacement_1s", # "a_long_std_1s", "is_special_lane", "can_go_right", "can_go_left",
        "TTC", "actual_gap", "rel_speed" #"traffic_pressure", , "gap_rate_trend_1s", "gap_rate_of_change", 
    ]
    
    # 2. Train/Test Split
    # Note: Hard mining can be integrated into data_split_with_sampling directly if desired.
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, sampling_keep_factor=4)

    if not check_data_leakage(df, X_train, X_test):
        print("Data leakage detected! Check vehicle splits.")
        return

    # 3. Model Training
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=15,
        min_samples_leaf=10,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 4. Feature Importance
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    importances.plot(kind='barh', color='skyblue')
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

    # 5. Prediction & Smoothing
    # We apply smoothing to the probabilities to remove frame-by-frame 'jitter'
    y_probs_raw = clf.predict_proba(X_test)

    # Post-process probabilities before thresholding
    # Slightly penalize Right predictions to force higher certainty
    #y_probs_raw[:, 2] = y_probs_raw[:, 2] * 0.9  # 10% 'Skepticism' factor for Right turns
    
    # Map back to Vehicle IDs to smooth within individual timelines
    # test_results = df.loc[X_test.index, ['Vehicle_Global_ID', 'Frame_Global_ID', 'Location']].copy()
    # test_results['actual_class'] = y_test
    # test_results['prob_left'] = y_probs_raw[:, 1]
    # test_results['prob_right'] = y_probs_raw[:, 2]

    # # Smooth probabilities (Rolling window of 0.5s / 5 frames)
    # test_results['prob_left_smooth'] = test_results.groupby('Vehicle_Global_ID')['prob_left'].transform(
    #     lambda x: x.rolling(5, center=True).mean()).fillna(test_results['prob_left'])
    # test_results['prob_right_smooth'] = test_results.groupby('Vehicle_Global_ID')['prob_right'].transform(
    #     lambda x: x.rolling(5, center=True).mean()).fillna(test_results['prob_right'])

    # # 6. Threshold Optimization
    # y_probs_smooth = test_results[['prob_left_smooth', 'prob_right_smooth']].values
    # # Note: Binary check for optimization
    # thresh_left, f1_left = find_best_threshold((y_test == 1).astype(int), test_results['prob_left_smooth'])
    # thresh_right, f1_right = find_best_threshold((y_test == 2).astype(int), test_results['prob_right_smooth'])

    # 2. Find the optimal thresholds for Class 1 and Class 2
    thresh_left, f1_left = find_best_threshold((y_test == 1).astype(int), y_probs_raw[:, 1])
    thresh_right, f1_right = find_best_threshold((y_test == 2).astype(int), y_probs_raw[:, 2])

    print(f"Optimal Threshold (Left):  {thresh_left:.4f} | Max F1: {f1_left:.4f}")
    print(f"Optimal Threshold (Right): {thresh_right:.4f} | Max F1: {f1_right:.4f}")

    # 7. Final Evaluation
    y_pred_opt = apply_custom_thresholds(y_probs_raw, thresh_left, thresh_right)
    print("\n--- Optimized Classification Report ---")
    print(classification_report(y_test, y_pred_opt, target_names=['None', 'Left', 'Right']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    ConfusionMatrixDisplay(cm, display_labels=['None', 'Left', 'Right']).plot(cmap='Blues')
    plt.show()

    # 8. Anticipation Plot
    # window_size = 30
    # time_steps = np.arange(-window_size, 1) * 0.1
    # avg_left = get_avg_prob_curve(test_results, 1, 'prob_left_smooth', window_size)
    # avg_right = get_avg_prob_curve(test_results, 2, 'prob_right_smooth', window_size)

    # plt.figure(figsize=(10, 5))
    # if avg_left is not None:
    #     plt.plot(time_steps, avg_left, label='Mean Left Prob', color='blue', lw=2)
    # if avg_right is not None:
    #     plt.plot(time_steps, avg_right, label='Mean Right Prob', color='red', lw=2)
    
    # plt.axhline(y=thresh_left, color='blue', ls='--', alpha=0.4, label='Thresh Left')
    # plt.axhline(y=thresh_right, color='red', ls='--', alpha=0.4, label='Thresh Right')
    # plt.title("Anticipation Curve (Smoothed Probabilities)")
    # plt.xlabel("Seconds Before Lane Change")
    # plt.ylabel("Probability")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

if __name__ == "__main__":
    main()