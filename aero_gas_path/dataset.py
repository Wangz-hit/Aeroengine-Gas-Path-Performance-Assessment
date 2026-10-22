"""Chronological train/validation/test partitions for rolling windows."""

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def chronological_split(n: int, train_fraction: float = 0.7,
                        valid_fraction: float = 0.15) -> TemporalSplit:
    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1-train_fraction:
        raise ValueError("Invalid train/validation fractions.")
    train_end = int(n*train_fraction)
    valid_end = int(n*(train_fraction+valid_fraction))
    if min(train_end, valid_end-train_end, n-valid_end) < 2:
        raise ValueError("Each chronological partition needs at least two records.")
    return TemporalSplit(np.arange(0, train_end),
                         np.arange(train_end, valid_end),
                         np.arange(valid_end, n))


def rolling_windows(features: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or length < 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected an [N,3] array and length >= 2.")
    padded = np.pad(values, ((length-1, 0), (0, 0)), mode="edge")
    windows = np.stack([padded[i:i+length] for i in range(len(values))])
    return windows


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, indices: np.ndarray,
                 labels: np.ndarray | None = None):
        self.windows = torch.as_tensor(windows[indices], dtype=torch.float32)
        self.targets = self.windows[:, -1, :].clone()
        self.labels = None
        if labels is not None:
            if labels.shape != (len(windows), 5) or not np.isfinite(labels[indices]).all():
                raise ValueError("Expected finite [N,5] component labels.")
            self.labels = torch.as_tensor(labels[indices], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"windows": self.windows[index], "target": self.targets[index]}
        if self.labels is not None:
            item["label"] = self.labels[index]
        return item
