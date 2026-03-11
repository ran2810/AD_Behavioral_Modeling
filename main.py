import warnings
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
import numpy as np

# Local imports
from data.preprocess import preprocess_all
from data.create_lstm_dataset import LaneChangeSequenceDataset
from extract_features import calculate_features
from utils.data_prep import data_split_with_sampling, check_data_leakage
from models.rfclassifier import LaneChangeClassifier
from models.lstm import LaneChangeLSTM


# Suppress all FutureWarnings globally
warnings.simplefilter(action='ignore', category=FutureWarning)

def trigger_rf(df, features):

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

def validate_lstm(model, loader, device, thresh_left, thresh_right):
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
    
    # Reuse your existing apply_custom_thresholds logic
    preds = apply_custom_thresholds(probs, thresh_left, thresh_right)
    
    f1_left = f1_score(labels, preds, labels=[1], average='macro')
    f1_right = f1_score(labels, preds, labels=[2], average='macro')
    
    return f1_left, f1_right

def  trigger_lstm(df, features):

    # Train/Test Split
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, sampling_keep_factor=4)
        
    window_size = 30 # 3 seconds of history at 10Hz
    train_dataset = LaneChangeSequenceDataset(X_train, y_train, df.loc[X_train.index, 'Vehicle_Global_ID'], window_size)
    test_dataset = LaneChangeSequenceDataset(X_test, y_test, df.loc[X_test.index, 'Vehicle_Global_ID'], window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneChangeLSTM(input_size=len(features), hidden_size=64, num_layers=2).to(device)
    
    # Use the optimized weights: {None: 1, Left: 3, Right: 10}
    weights = torch.tensor([1.0, 3.0, 10.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    print(f"Training LSTM on {device}...")
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    # 4. Evaluation
    # Note: Use model.predict_proba equivalents for threshold optimization logic
    # established in the RF trials.
    #validate_lstm(model, test_loader, device, )

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='provide the model name to trigger')
    parser.add_argument('model_name', type=str,
                        help='rf or lstm or both')
    args = parser.parse_args()

    # Ensure features include the best set: v_lat, Lane_ID, and Target Gaps
    # Data Loading and Feature Engineering
    DATA_PATH = r"H:\GitHub\AD_Behavioral_Modeling\data"
    df = preprocess_all(DATA_PATH)
    df = calculate_features(df)
    
    features = [
        "v_vel", "a_long", "v_lat", "Lane_ID", "v_lat_lag_5", "v_lat_lag_10", 
        "lat_displacement_1s", "can_go_right", "can_go_left", "a_long_std_1s", 
        "TTC", "actual_gap", "gap_rate_trend_1s", "rel_speed"
    ]

    if args.model_name == "rf":
        trigger_rf(df, features)

    if args.model_name == "lstm":
        trigger_lstm(df, features)
    
    if args.model_name == "both":
        trigger_rf(df, features)
        trigger_lstm(df, features)