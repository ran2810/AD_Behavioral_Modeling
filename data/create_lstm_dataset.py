import torch
from torch.utils.data import Dataset
import numpy as np


class LaneChangeSequenceDataset(Dataset):
    """
    Lazy sequence dataset — stores only start indices, not pre-materialized sequences.

    Memory cost:
      Old approach: N_seq x window x features x 4 bytes  
      New approach: N_seq x 4 bytes (start idx) + X_values ref
    """

    def __init__(self, X, y, vehicle_ids, window_size=30):
        self.window_size = window_size

        # Store feature matrix as float32 — halves memory vs default float64
        self.X_values = X.values.astype(np.float32)
        y_values      = y.values
        veh_values    = vehicle_ids.values

        print("Generating sequence indices... (respecting vehicle boundaries)")

        starts = []
        labels = []

        # iterate over each vehicle — windows must stay within one vehicle's trajectory
        for veh in np.unique(veh_values):
            # positions of this vehicle's rows in the sorted X array
            veh_pos = np.where(veh_values == veh)[0]

            # skip vehicles that are shorter than one window — can't form a sequence
            if len(veh_pos) < window_size:
                continue

            # slide the window one frame at a time along this vehicle's trajectory
            for i in range(len(veh_pos) - window_size + 1):
                window = veh_pos[i: i + window_size]

                # positions must be consecutive integers — if not, there's a gap caused by
                # undersampling or a data edge; skip to avoid mixing non-adjacent frames
                if window[-1] - window[0] != window_size - 1:
                    continue

                # record the start position and the label at the last frame of the window
                starts.append(window[0])
                labels.append(y_values[window[-1]])

        self.sequence_starts = np.array(starts, dtype=np.int32)
        self.labels          = np.array(labels, dtype=np.int32)

        # Inverse-frequency weights — stored as a single 1D tensor, not a list of tensors.
        # The list comprehension version creates 6M+ individual Python objects (~1.3 GB).
        class_counts        = np.bincount(self.labels)
        weights_per_class   = (1.0 / class_counts).astype(np.float32)
        self.sample_weights = torch.from_numpy(weights_per_class[self.labels])

        print(f"  Total sequences: {len(self.labels):,}  "
              f"(None={int((self.labels==0).sum()):,}  "
              f"Left={int((self.labels==1).sum()):,}  "
              f"Right={int((self.labels==2).sum()):,})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = int(self.sequence_starts[idx])
        # Slice is a view; .copy() makes it contiguous for torch
        seq = self.X_values[start: start + self.window_size].copy()
        return (
            torch.from_numpy(seq),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )
