"""
Transform bare-bones

Reads: data/ccr_raw.pkl
Writes: data/X_train.npy, data/X_val.npy, data/y_train.npy, data/y_val.npy
        data/scaler.pkl, data/feature_columns.json

Run: python -m src.transform_class_vibe
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.paths import (
    RAW_PKL_PATH,
    X_TRAIN_PATH, X_VAL_PATH, Y_TRAIN_PATH, Y_VAL_PATH,
    SCALER_PATH, FEATURES_PATH
)

# --- feature/target contract

FEATURE_COLUMNS: List[str] = [
    "gpa",
    "attendance_rate",
    "sat_math",
    "sat_ebrw",
    "cte_hours",
    "internship",
    "career_interest_clarity",
    "self_efficacy",
]

TARGET_COLUMN = "readiness_score"

@dataclass(frozen=True)
class TransformConfig:
    test_size: float = 0.2
    seed: int = 42

class CCRTransformer:
    def __init__(self, *, feature_cols: List[str], target_col: str) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()

    def fit_transform_split(self, df: pd.DataFrame, *, config: TransformConfig) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # 1) Slice
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy(dtype=np.float32)

        # 2) split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.test_size, random_state=config.seed)

        # 3) scale, fit on train only
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)

        return X_train, X_val, y_train, y_val


def main() -> None:
    if not RAW_PKL_PATH.exists():
        raise FileNotFoundError(f"Missing raw artifact: {RAW_PKL_PATH}\n"
                                "Run: python -m src.extract_class_vibe")

    df = pd.read_pickle(RAW_PKL_PATH)

    transformer = CCRTransformer(feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)
    cfg = TransformConfig(test_size=0.2, seed=42)

    X_train, X_val, y_train, y_val = transformer.fit_transform_split(df, config=cfg)

    # Save arrays for the load step
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_VAL_PATH, X_val)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_VAL_PATH, y_val)

    # Save artifacts
    pd.to_pickle(transformer.scaler, SCALER_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")

    print("Saved transform artifacts:")
    print(f"  {X_TRAIN_PATH.name}:  {X_train.shape}")
    print(f"  {X_VAL_PATH.name}:    {X_val.shape}")
    print(f"  {Y_TRAIN_PATH.name}:  {y_train.shape}")
    print(f"  {Y_VAL_PATH.name}:    {y_val.shape}")
    print(f"  {SCALER_PATH.name}")
    print(f"  {FEATURES_PATH.name}")

if __name__ == "__main__":
    main()




