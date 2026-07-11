"""Download and audit the UEA FaceDetection dataset without modifying its values."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DATASET = "FaceDetection"
DEFAULT_DATA_DIR = Path("data/raw")


def load_split(dataset: str, split: str, data_dir: Path, download: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one archive split, optionally allowing aeon to download it locally."""
    try:
        from aeon.datasets import load_classification
    except ImportError as error:
        raise SystemExit(
            "aeon is not installed. Activate the project environment and run "
            '`python -m pip install -e ".[dev]"`.'
        ) from error

    dataset_path = data_dir / dataset
    if not download and not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found at {dataset_path}. Re-run with --download to fetch the public archive."
        )

    X, y, metadata = load_classification(
        name=dataset,
        split=split,
        extract_path=data_dir,
        return_metadata=True,
    )
    if not isinstance(X, np.ndarray) or X.ndim != 3:
        raise ValueError(f"Expected an equal-length 3D array; received {type(X)!r} with shape {getattr(X, 'shape', None)}.")
    return X, y, metadata


def describe_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """Print audit facts needed before preprocessing or modelling."""
    counts = dict(sorted(Counter(map(str, y)).items()))
    channel_mean = X.mean(axis=(0, 2))
    channel_std = X.std(axis=(0, 2))

    print(f"{name} shape (examples, channels, time): {X.shape}")
    print(f"{name} labels: {counts}")
    print(f"{name} finite values: {np.isfinite(X).all()}")
    print(
        f"{name} channel means: min={channel_mean.min():.4f}, "
        f"median={np.median(channel_mean):.4f}, max={channel_mean.max():.4f}"
    )
    print(
        f"{name} channel stds: min={channel_std.min():.4f}, "
        f"median={np.median(channel_std):.4f}, max={channel_std.max():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="Allow aeon to download a missing dataset.")
    args = parser.parse_args()

    for split in ("train", "test"):
        X, y, metadata = load_split(args.dataset, split, args.data_dir, args.download)
        describe_split(split, X, y)
        print(f"{split} metadata: {metadata}")


if __name__ == "__main__":
    main()
