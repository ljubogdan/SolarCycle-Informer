import torch
from torch.utils.data import Dataset
import numpy as np

class SunspotDataset(Dataset):
    def __init__(self, data, context_length=60, prediction_length=12):
        self.data = data.values.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length

    def __len__(self):
        return max(1, len(self.data) - self.total_length + 1)

    def __getitem__(self, idx):

        sequence = self.data[idx:idx + self.total_length]
        past_values = sequence[:self.context_length]
        future_values = sequence[self.context_length:self.total_length]

        return {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "future_values": torch.tensor(future_values, dtype=torch.float32)
        }