import torch
from torch.utils.data import Dataset
import numpy as np


class LaneChangeSequenceDataset(Dataset):
    def __init__(self, X, y, vehicle_ids, window_size=30):
        """
        Args:
            X:           Pandas DataFrame of features — must be sorted by vehicle + frame
            y:           Pandas Series of labels (0, 1, 2)
            vehicle_ids: Pandas Series of Global Vehicle IDs (same index as X/y)
            window_size: Number of consecutive past frames used as input history
        """
        self.sequences = []
        self.labels = []

        X_values   = X.values
        y_values   = y.values
        veh_values = vehicle_ids.values

        print("Generating sequences... (respecting vehicle boundaries)")

        for veh in np.unique(veh_values):
            # Positions of this vehicle's rows within the (sorted) X array
            veh_pos = np.where(veh_values == veh)[0]

            if len(veh_pos) < window_size:
                continue

            # Sliding window: gather exactly window_size consecutive positions
            for i in range(len(veh_pos) - window_size + 1):
                window = veh_pos[i: i + window_size]

                # Guard: positions must be consecutive — no gaps from undersampling
                if window[-1] - window[0] != window_size - 1:
                    continue

                self.sequences.append(X_values[window])          # shape (window_size, n_features)
                self.labels.append(y_values[window[-1]])          # label at the end of the window

        # Inverse-frequency sample weights so the DataLoader sees rare classes more often
        class_counts  = np.bincount(self.labels)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        self.sample_weights = [class_weights[label] for label in self.labels]

        print(f"  Total sequences: {len(self.labels):,}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )
