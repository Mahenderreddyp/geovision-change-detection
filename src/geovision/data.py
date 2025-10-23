from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]  # (6, H, W) = t0(3)+t1(3)
        y = self.y[idx]  # (1, H, W)
        half = x.shape[0] // 2
        t0 = torch.tensor(x[:half], dtype=torch.float32)
        t1 = torch.tensor(x[half:], dtype=torch.float32)
        y  = torch.tensor(y, dtype=torch.float32)
        return t0, t1, y