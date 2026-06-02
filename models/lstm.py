import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class LaneChangeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(LaneChangeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # dropout between LSTM layers (only applied when num_layers > 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # dropout on the last hidden state before classification
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # h0/c0 default to zeros when not passed — no need to create them explicitly
        out, _ = self.lstm(x)

        # take the last time-step hidden state
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

    def save_model(self, filepath, features, thresholds):
        """Save weights + feature list + thresholds in one file (mirrors RF convention)."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'model_state':  self.state_dict(),
            'hidden_size':  self.hidden_size,
            'num_layers':   self.num_layers,
            'input_size':   self.fc.in_features,
            'features':     features,
            'thresholds':   thresholds,
        }, filepath)
        print(f"LSTM saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Load a saved LSTM and return (model, features, thresholds)."""
        data    = torch.load(filepath, map_location='cpu')
        model   = cls(input_size=data['input_size'],
                      hidden_size=data['hidden_size'],
                      num_layers=data['num_layers'])
        model.load_state_dict(data['model_state'])
        print(f"LSTM loaded from {filepath}")
        return model, data['features'], data['thresholds']

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['None', 'Left', 'Right']).plot(cmap='Blues')
        plt.title("LSTM Confusion Matrix")
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"CM saved to {save_path}")
        plt.show()
        plt.close()

    def plot_feature_importance(self, importances, features, save_path=None):
        """Permutation importance: higher = more important (F1 drop when feature is shuffled)."""
        series = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        plt.figure(figsize=(8, 5))
        plt.barh(list(series.keys()), list(series.values()), color='steelblue')
        plt.xlabel("Mean F1 drop (permutation importance)")
        plt.title("LSTM Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Feature importance saved to {save_path}")
        plt.show()
        plt.close()
