"""
HARBench Dataset

PyTorch Dataset class for finetuning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    """
    Human Activity Recognition Dataset

    Args:
        X: Sensor data (N, C, T) - N: number of samples, C: number of channels, T: sequence length
        Y: Labels (N,)
        transform: Optional preprocessing function
    """

    def __init__(self, X, Y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
