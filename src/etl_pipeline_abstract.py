"""
- Extract: load raw tabular data, source specified at runtime
- Transform: clean + feature engineer + split
- Load: adapt into PyTorch Dataset/DataLoader

Run: python etl_pipeline_abstract.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, Protocol, Sequence, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader


# 1) Define Runtime Contracts

class Extractor(Protocol):

    def extract(self, *, source: Any, **kwargs) -> pd.DataFrame:
        """Return raw DataFrame from any source"""
        ...

class Transformer(Protocol):

    def fit_transform(self, df: pd.DataFrame, *, config: "TransformConfig") -> Tuple[np.ndarray, np.ndarray, "TransformArtifacts"]:
        """Fit preprocessing on df and return X, y, and artifacts"""
        ...

    def transform(self, df: pd.DataFrame, *, artifacts: "TransformArtifacts", config: "TransformConfig") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply preprocessing with existing artifacts. Returns y if present"""
        ...

class Loader(Protocol):

    def load(self, X: np.ndarray, y: np.ndarray, *, config: "LoadConfig") -> "LoadedData":
        """Package arrays into PyTorch Dataset/DataLoader."""
        ...



# 2) Config + Artifacts

@dataclass(frozen=True)
class TransformConfig:
    target_col: str
    numeric_cols: Optional[Sequence[str]] = None
    categorical_cols: Optional[Sequence[str]] = None
    drop_cols: Optional[Sequence[str]] = None
    impute_numeric: str = "median"
    one_hot_drop_first: bool = True


@dataclass
class TransformArtifacts:
    scaler: StandardScaler
    feature_columns: List[str]

@dataclass(frozen=True)
class LoadConfig:
    batch_size: int = 64
    test_size: float = 0.2
    seed: int = 42
    shuffle_train: bool = True
    num_workers: int = 0


@dataclass
class LoadedData:
    train_loader: DataLoader
    val_loader: DataLoader
    artifacts: TransformArtifacts


# 3) Implementations

class CSVExtractor:
    def extract(self, *, source: Any, **kwargs) -> pd.DataFrame:
        """
        :param source: str path to CSV
        :param kwargs: passed to pd.read_csv
        :return pd.read_csv: pandas DataFrame
        """
        if not isinstance(source, str) or not source:
            raise ValueError("CSVExtractor expects source=<csv_path: str>.")
        return pd.read_csv(source, **kwargs)


class SklearnTabularTransformer:
    """Default transformer"""

    def fit_transform(self, df: pd.DataFrame, *, config: TransformConfig) -> Tuple[np.ndarray, np.ndarray, TransformArtifacts]:

        df = self._prepare_df(df, config=config)

        if config.target_col not in df.columns:
            raise ValueError(f"target_col '{config.target_col}' not found in DataFrame columns.")

        y = df[config.target_col].to_numpy(dtype=np.float32)
        X_df = df.drop(columns=[config.target_col])

        numeric_cols, categorical_cols = self._resolve_columns(X_df, config=config)

        # impute numeric if necessary

        X_df = self._impute_numeric(X_df, numeric_cols=numeric_cols, strategy=config.impute_numeric)

        # One-hot categorical cols
        if categorical_cols:
            X_df = pd.get_dummies(X_df, columns=list(categorical_cols), drop_first=config.one_hot_drop_first)

        feature_columns = X_df.columns.tolist()

        # Scale feature cols
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.to_numpy(dtype=np.float32)).astype(np.float32)

        artifacts = TransformArtifacts(scaler=scaler, feature_columns=feature_columns)

        return X, y, artifacts

    def transform(self, df: pd.DataFrame, *, artifacts: TransformArtifacts, config: TransformConfig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        df = self._prepare_df(df, config=config)

        y: Optional[np.ndarray] = None

        if config.target_col in df.columns:
            y = df[config.target_col].to_numpy(dtype=np.float32)
            X_df = df.drop(columns=[config.target_col])
        else:
            X_df = df

        numeric_cols, categorical_cols = self._resolve_columns(X_df, config=config)

        X_df = self._impute_numeric(X_df, numeric_cols=numeric_cols, strategy=config.impute_numeric)

        if categorical_cols:
            X_df = pd.get_dummies(X_df, columns=list(categorical_cols), drop_first=config.one_hot_drop_first)


        # align to training feature schema
        for col in artifacts.feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0

        X_df = X_df[artifacts.feature_columns]
        X = artifacts.scaler.transform(X_df.to_numpy(dtype=np.float32)).astype(np.float32)

        return X, y

    def _prepare_df(self, df: pd.DataFrame, *, config: TransformConfig) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Transformer expects a pandas DataFrame.")
        df = df.copy()

        if config.drop_cols:
            drop_existing = [c for c in config.drop_cols if c in df.columns]
            if drop_existing:
                df = df.drop(columns=drop_existing)

        return df

    def _resolve_columns(self, X_df: pd.DataFrame, *, config: TransformConfig) -> Tuple[Sequence[str], Sequence[str]]:

        if config.numeric_cols is None:
            numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [c for c in config.numeric_cols if c in X_df.columns]

        if config.categorical_cols is None:
            categorical_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        else:
            categorical_cols = [c for c in config.categorical_cols if c in X_df.columns]

        return numeric_cols, categorical_cols

    def _impute_numeric(self, X_df: pd.DataFrame, *, numeric_cols: Sequence[str], strategy: str) -> pd.DataFrame:
        if not numeric_cols:
            return X_df


        X_df = X_df.copy()

        for col in numeric_cols:
            if X_df[col].isna().any():
                if strategy == "median":
                    fill = float(X_df[col].median())
                elif strategy == "mean":
                    fill = float(X_df[col].mean())
                else:
                    raise ValueError("impute_numeric must be median or mean.")
                X_df[col] = X_df[col].fillna(fill)
        return X_df

class TabularRegressionDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).reshape(-1, 1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TorchDataLoaderBuilder:

    def load(self, X: np.ndarray, y: np.ndarray, *, config: LoadConfig) -> LoadedData:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.test_size, random_state=config.seed)

        train_ds = TabularRegressionDataset(X_train, y_train)
        val_ds = TabularRegressionDataset(X_val, y_val)

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=config.shuffle_train,
            num_workers=config.num_workers,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        # Artifacts are attached by pipeline, loader should not care about artifacts.
        # Artifacts will be set later.
        return LoadedData(train_loader=train_loader, val_loader=val_loader, artifacts=TransformArtifacts(StandardScaler(), []))

# 4) The Actual Pipeline (Core)

class ETLPipeline:

    def __init__(self, *, extractor: Extractor, transformer: Transformer, loader: Loader) -> None:
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader

    def run(
            self, *, extract_source: Any, extract_kwargs: Optional[Dict[str, Any]] = None,
            transform_config: TransformConfig, load_config: LoadConfig) -> LoadedData:
        extract_kwargs = extract_kwargs or {}

        raw_df = self.extractor.extract(source=extract_source, **extract_kwargs)
        X, y, artifacts = self.transformer.fit_transform(raw_df, config=transform_config)

        loaded = self.loader.load(X, y, config=load_config)
        loaded.artifacts = artifacts
        return loaded


# 5) Entry Point with runtime configuration

def main() -> None:

    # Decide concrete implementations of config
    extractor = CSVExtractor()
    transformer = SklearnTabularTransformer()
    loader = TorchDataLoaderBuilder()

    etl = ETLPipeline(extractor=extractor, transformer=transformer, loader=loader)

    # specify behavior via config, not hardcoded inside classes
    transform_config = TransformConfig(
        target_col = "column_name",
        drop_cols=None,
        numeric_cols=None,
        categorical_cols=None,
        impute_numeric="median",
        one_hot_drop_first=True,
    )

    load_config = LoadConfig(batch_size=16, test_size=0.2, seed=42)

    # Set csv path here in main
    csv_path = "your_dataset.csv"

    loaded = etl.run(
        extract_source=csv_path,
        extract_kwargs=None,
        transform_config=transform_config,
        load_config=load_config,
    )

    xb, yb = next(iter(loaded.train_loader))
    print("Batch X:", xb.shape, "Batch y:", yb.shape)
    print("Num features:", xb.shape[1])
    print("First 8 feature names:", loaded.artifacts.feature_columns[:8])

if __name__ == "__main__":
    main()
