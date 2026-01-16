"""


"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.paths import(
    X_TRAIN_PATH, X_VAL_PATH, Y_TRAIN_PATH,
    Y_VAL_PATH, TRAIN_LOADER_INFO_PATH, VAL_LOADER_INFO_PATH
)

class NumpyRegressionDataset(Dataset):
    """Returns x_i, y_i as tensors"""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().reshape(-1, 1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

@dataclass(frozen=True)
class LoadConfig:
    batch_size: int = 32
    shuffle_train: bool = True
    num_workers: int = 0


class CCRLoader:
    def __init__(self, *, config: LoadConfig) -> None:
        self.config = config

    def build(self) -> Tuple[DataLoader, DataLoader]:

        # 1) Load transformed arrays
        X_train = np.load(X_TRAIN_PATH)
        X_val = np.load(X_VAL_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        y_val = np.load(Y_VAL_PATH)


        # 2) Wrap arrays as tensor datasets
        train_ds = NumpyRegressionDataset(X_train, y_train)
        val_ds = NumpyRegressionDataset(X_val, y_val)

        # 3) Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
        )
        return train_loader, val_loader

def main() -> None:
    cfg = LoadConfig(batch_size=32, shuffle_train=True, num_workers=0)
    loader = CCRLoader(config=cfg)

    train_loader, val_loader = loader.build()


    # Sanity check
    xb, yb = next(iter(train_loader))
    print("Train batch shapes:  ", xb.shape, yb.shape)

    xb2, yb2 = next(iter(val_loader))
    print("Val batch shapes:    ", xb2.shape, yb2.shape)

    # Optional: for viewing and learning purposes save training info metadata as json
    train_info = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle_train": cfg.shuffle_train,
        "num_batches": len(train_loader),
        "example_batch_X_shape": list(xb.shape),
        "example_batch_y_shape": list(yb.shape),
    }
    val_info = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "shuffle_train": False,
        "num_batches": len(val_loader),
        "example_batch_X_shape": list(xb2.shape),
        "example_batch_y_shape": list(yb2.shape),
    }

    TRAIN_LOADER_INFO_PATH.write_text(json.dumps(train_info, indent=2), encoding="utf-8")
    VAL_LOADER_INFO_PATH.write_text(json.dumps(val_info, indent=2), encoding="utf-8")
    print("Saved loader metadata artifacts.")


if __name__ == "__main__":
    main()








