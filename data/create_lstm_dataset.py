import torch
from torch.utils.data import Dataset

class LaneChangeSequenceDataset(Dataset):
    def __init__(self, X, y, vehicle_ids, window_size=30):
        self.sequences = []
        self.labels = []
        
        # Group by vehicle to ensure sequences are temporally continuous
        unique_vehs = vehicle_ids.unique()
        for veh in unique_vehs:
            veh_data = X[vehicle_ids == veh].values
            veh_labels = y[vehicle_ids == veh].values
            
            if len(veh_data) < window_size:
                continue
                
            # Create sliding windows
            for i in range(len(veh_data) - window_size):
                self.sequences.append(veh_data[i : i + window_size])
                # Label is the intent at the final frame of the window
                self.labels.append(veh_labels[i + window_size - 1])
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.long))