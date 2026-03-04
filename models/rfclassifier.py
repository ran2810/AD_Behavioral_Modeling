import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_recall_curve)

class LaneChangeClassifier:
    def __init__(self, n_estimators=100, max_depth=15, custom_weights=None, random_state=42):
        self.custom_weights = custom_weights or {0: 1.0, 1: 3.0, 2: 10.0}
        self.features = [] # To be stored upon saving
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=self.custom_weights,
            max_depth=max_depth,
            min_samples_leaf=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=random_state
        )
        self.thresh_left = 0.5
        self.thresh_right = 0.5

    def fit(self, X_train, y_train):
        """Trains the Random Forest model."""
        self.features = X_train.columns.tolist()
        self.clf.fit(X_train, y_train)

    def _find_best_threshold(self, y_true, y_probabilities):
        """Finds the threshold that maximizes the F1-score for a specific class."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probabilities)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[min(best_idx, len(thresholds)-1)], f1_scores[best_idx]

    def optimize_thresholds(self, X_test, y_test):
        """Calculates and stores optimal thresholds for Left and Right changes."""
        probs = self.clf.predict_proba(X_test)
        
        self.thresh_left, f1_l = self._find_best_threshold((y_test == 1).astype(int), probs[:, 1])
        self.thresh_right, f1_r = self._find_best_threshold((y_test == 2).astype(int), probs[:, 2])
        
        print(f"Optimal Threshold (Left):  {self.thresh_left:.4f} | Max F1: {f1_l:.4f}")
        print(f"Optimal Threshold (Right): {self.thresh_right:.4f} | Max F1: {f1_r:.4f}")
        return probs

    def predict_optimized(self, probs):
        """Applies stored thresholds to raw probabilities."""
        final_preds = np.zeros(len(probs))
        for i in range(len(probs)):
            p_left, p_right = probs[i, 1], probs[i, 2]
            if p_left >= self.thresh_left and p_left > p_right:
                final_preds[i] = 1
            elif p_right >= self.thresh_right and p_right > p_left:
                final_preds[i] = 2
            else:
                final_preds[i] = 0
        return final_preds

    def save_model(self, filepath="lane_change_model.joblib"):
        """Saves the model, thresholds, and feature list to a single file."""
        model_data = {
            'classifier': self.clf,
            'thresholds': {
                'left': self.thresh_left,
                'right': self.thresh_right
            },
            'features': self.features
        }
        joblib.dump(model_data, filepath)
        print(f"Model and thresholds saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Class method to load a saved model and return a LaneChangeClassifier instance."""
        data = joblib.load(filepath)
        instance = cls()
        instance.clf = data['classifier']
        instance.thresh_left = data['thresholds']['left']
        instance.thresh_right = data['thresholds']['right']
        instance.features = data['features']
        print(f"Model loaded from {filepath}")
        return instance

    def plot_importance(self):
        """Visualizes feature importance."""
        importances = pd.Series(self.clf.feature_importances_, index=self.features).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        importances.plot(kind='barh', color='skyblue')
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred):
        """Plots the confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['None', 'Left', 'Right']).plot(cmap='Blues')
        plt.show()