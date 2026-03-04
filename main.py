import warnings

# Local imports
from data.preprocess import preprocess_all
from extract_features import calculate_features
from utils.data_prep import data_split_with_sampling, check_data_leakage
from models.rfclassifier import LaneChangeClassifier
from sklearn.metrics import classification_report

# Suppress all FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # Data Loading and Feature Engineering
    DATA_PATH = r"H:\GitHub\AD_Behavioral_Modeling\data"
    df = preprocess_all(DATA_PATH)
    df = calculate_features(df)
    
    features = [
        "v_vel", "a_long", "v_lat", "Lane_ID", "v_lat_lag_5", "v_lat_lag_10", 
        "lat_displacement_1s", "can_go_right", "can_go_left", "a_long_std_1s", 
        "TTC", "actual_gap", "gap_rate_trend_1s", "rel_speed"
    ]

    # Train/Test Split
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, sampling_keep_factor=4)

    # Check data leakage
    if not check_data_leakage(df, X_train, X_test):
        print("Data leakage detected! Check vehicle splits.")
        return

    # Initialize and Train Classifier 
    custom_weights = {0: 1.0, 1: 3.0, 2: 10.0}
    model = LaneChangeClassifier(custom_weights=custom_weights)
    
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Visualization & Threshold Optimization
    model.plot_importance()
    y_probs = model.optimize_thresholds(X_test, y_test)

    # Final Evaluation
    y_pred_opt = model.predict_optimized(y_probs)
    
    print("\n--- Optimized Classification Report ---")
    print(classification_report(y_test, y_pred_opt, target_names=['None', 'Left', 'Right']))

    # Final Plot
    model.plot_confusion_matrix(y_test, y_pred_opt)

    # save model
    model.save_model("best_rf_model_v1.joblib")


if __name__ == "__main__":
    main()