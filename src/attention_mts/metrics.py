"""Shared evaluation metrics so every model in the comparison is scored identically"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    auroc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)
    

def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    """y_true: (N, ) int labels in {0,1}. y_prob: (N,) predicted probability of class 1."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        macro_f1=f1_score(y_true, y_pred, average="macro"),
        auroc=roc_auc_score(y_true, y_prob),
    )


def count_parameters(model: torch.nn.Module)-> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: torch.nn.Module, example_input: torch.Tensor, n_repeats: int = 50
) -> float:
    """Mean forward-pass latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(5): # warm-up, excluded from the measurment
            model(example_input)
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(example_input)
        elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) * 1000.0
