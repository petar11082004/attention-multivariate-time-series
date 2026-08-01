"""Loading, label encoding, and per-fold standardisation for the FaceDetection dataset.

Input convention used throughout the project: X.shape == (n_examples, n_channels, n_time_steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATASET = "FaceDetection"
DEFAULT_DATA_DIR = Path("data/raw")


def load_split(
    dataset: str = DEFAULT_DATASET,
    split: str = "train",
    data_dir: Path = DEFAULT_DATA_DIR,
    download: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one archive split as (X, y) with X float32 and y int64 in {0,1}."""
    from aeon.datasets import load_classification

    dataset_path = data_dir / dataset
    if not download and not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path}. Re-run with download=True to fetch the public archive."
        )

    X, y = load_classification(name=dataset, split=split, extract_path=data_dir)
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError(
            f"Expected an equal-length 3D array; received {type(X)!r} with shape {getattr(X, 'shape', None)}"
        )
    return X.astype(np.float32), y.astype(np.int64)


@dataclass
class ChannelStandardiser:
    """Per-channel z-score. Fit only on a training fold, then transform any fold with it"""
    
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ChannelStandardiser":
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True)
        self.std[self.std < 1e-8] = 1e-8
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ChannelStandardiser must be fit before transform.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)