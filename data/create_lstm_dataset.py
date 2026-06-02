import torch
from torch.utils.data import Dataset
import numpy as np

class LaneChangeSequenceDataset(Dataset):
    def __init__(self, X, y, vehicle_ids, window_size=30):
        """
        Args:
            X: Pandas DataFrame of features
            y: Pandas Series of labels (0, 1, 2)
            vehicle_ids: Pandas Series of Global Vehicle IDs
            window_size: Number of past frames (history) for the LSTM
        """
        self.sequences = []
        self.labels = []
        self.sample_weights = []

        # Use indices to avoid duplicating massive DataFrames in memory
        X_values = X.values
        y_values = y.values
        veh_values = vehicle_ids.values
        
        print("Generating sequences... (respecting vehicle boundaries)")

        unique_vehs = np.unique(veh_values)
        for veh in unique_vehs:
            # Get indices for this specific vehicle
            veh_indices = np.where(veh_values == veh)[0]
            
            if len(veh_indices) < window_size:
                continue
                
            # Sliding window within the vehicle's trajectory
            for i in range(len(veh_indices) - window_size):
                # sequence: [i, i+1, ... i+49]
                # label: i+49 (the intent at the end of the history)
                start_idx = veh_indices[i]
                end_idx = veh_indices[i + window_size]
                
                self.sequences.append(X_values[start_idx:end_idx])
                self.labels.append(y_values[end_idx - 1])

        # Calculate Weights for Sampling (Inverse Frequency)
        # This helps the model "see" the Right class (14k frames) as often as the None class
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        self.sample_weights = [class_weights[label] for label in self.labels]
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.long))