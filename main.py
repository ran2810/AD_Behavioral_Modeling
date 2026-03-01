from data.preprocess import preprocess_all
from extract_features import calculate_features
from utils.sampling import data_split_with_sampling, check_data_leakage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # Load dataset
    df = preprocess_all(r"H:\GitHub\AD_Behavioral_Modeling\data")

    # Features & labels
    df = calculate_features(df)

    features = ["v_vel", "a_long", "Lane_ID", "v_lat", "v_lat_lag_5", "v_lat_lag_10", "TTC", "actual_gap", "rel_speed"]
    
    # Train/test split with sampling to reduce the 'boring - frames without lane changes' data
    X_train, X_test, y_train, y_test = data_split_with_sampling(df, features, 4)

    # check data leakage by vehicle ID 
    if check_data_leakage(df, X_train, X_test):

        # Train RF model
        clf = RandomForestClassifier(
        n_estimators=200,          # number of trees
        class_weight="balanced",
        max_depth=15,              # Prevents the trees from getting too deep and over-memorizing
        min_samples_leaf=10,       # Ensures each leaf has enough data to be statistically significant
        max_features='sqrt',       # Standard, but good to keep
        n_jobs=-1,                 # Use all CPU cores for speed
        verbose=2,                 # Prints progress updates
        random_state=42)

        clf.fit(X_train, y_train)


        # Get importance from the trained model
        importances = pd.Series(clf.feature_importances_, index=features)
        importances = importances.sort_values(ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        importances.plot(kind='barh', color='skyblue')
        plt.title("Feature Importance for Lane Change Prediction")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.gca().invert_yaxis() # Highest importance at the top
        plt.show()

        # # Evaluate - default threshold is 0.5
        #y_pred = clf.predict(X_test)
        #print(classification_report(y_test, y_pred))

        #Get probabilities instead of hard classes
        y_probs = clf.predict_proba(X_test)
        
        # Custom Thresholding: Only predict 1 or 2 if the model is > 70% sure
        # y_probs columns are [Class 0, Class 1, Class 2]
        custom_preds = np.zeros(len(y_test))
        for i in range(len(y_probs)):
           if y_probs[i, 1] > 0.7:
               custom_preds[i] = 1
           elif y_probs[i, 2] > 0.7:
               custom_preds[i] = 2
           else:
               custom_preds[i] = 0
        
        print(classification_report(y_test, custom_preds))

        cm = confusion_matrix(y_test, custom_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None', 'Left', 'Right'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

if __name__ == "__main__":
    main()
