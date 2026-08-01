"""Deterministic train/validation split utilities.

See docs/experiment-plan.md ("Validation split: known simplification") for why this is a
stratified random split rather than a participant-grouped split
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_train_val_split(
    y: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) into y, stratified by label."""
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, stratify=y, random_state=seed
    )
    return train_idx, val_idx

