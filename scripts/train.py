"""Train and evaluate any registered model architecture with shared framework."""

from __future__ import annotations

import argparse
import copy
from typing import Callable

import numpy as np
import torch
from torch import nn

from attention_mts.data import ChannelStandardiser, load_split
from attention_mts.metrics import compute_classification_metrics, count_parameters
from attention_mts.models import (
    BiLSTMBaseline,
    CNNBaseline,
    CNNTransformerHybrid,
    PooledLinearBaseline,
    TransformerClassifier,
)
from attention_mts.splits import stratified_train_val_split


SPLIT_SEED = 0 # fixed: every model/seed trains and validates on the same partition

ModelFactory = Callable[[int], nn.Module]

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "baseline": lambda n_channels: PooledLinearBaseline(n_channels=n_channels),
    "cnn": lambda n_channels: CNNBaseline(n_channels=n_channels, n_mid=48, n_hidden=96, kernel_size= 7),
    "transformer": lambda n_channels: TransformerClassifier(
    n_channels=n_channels, d_model=96, n_heads=4, d_ff=192, n_layers=2, dropout=0.1
),
    "bilstm": lambda n_channels: BiLSTMBaseline(n_channels=n_channels, hidden_size=96),
    "hybrid": lambda n_channels: CNNTransformerHybrid(
        n_channels=n_channels, n_mid=32, d_model=96, n_heads=4, d_ff=192, n_layers=2,
        kernel_size=7, downsample_factor=2, dropout=0.1,
    ),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_tensor(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(X), torch.from_numpy(y).float()


def prepare_data():
    X_train_full, y_train_full = load_split(split="train")
    train_idx, val_idx = stratified_train_val_split(y_train_full, val_fraction=0.2, seed=SPLIT_SEED)

    scaler = ChannelStandardiser().fit(X_train_full[train_idx])
    X_train = scaler.transform(X_train_full[train_idx])
    X_val = scaler.transform(X_train_full[val_idx])

    return to_tensor(X_train, y_train_full[train_idx]), to_tensor(X_val, y_train_full[val_idx])


def train_one_seed(model_name: str, seed: int, data, epochs: int, lr: float, batch_size: int, patience: int) -> dict:
    set_seed(seed)
    (X_train_t, y_train_t), (X_val_t, y_val_t) = data

    model = MODEL_FACTORIES[model_name](X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_train)
        epoch_loss= 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = permutation[start:start+batch_size]
            xb, yb = X_train_t[batch_idx], y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t).squeeze(-1), y_val_t).item()
        print(f"model={model_name} seed={seed} epoch={epoch + 1:02d} "
              f"train_loss={epoch_loss / n_train:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"model = {model_name} seed = {seed} early stop at epoch={epoch + 1:02d} "
                f"(best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)        
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_val_t).squeeze(-1)).numpy()
    metrics = compute_classification_metrics(y_val_t.numpy(), val_prob)
    return {
        "model": model_name, "seed": seed, "params": count_parameters(model),
        "best_val_loss": best_val_loss, **metrics.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), default="baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    data = prepare_data()
    results = [train_one_seed(args.model, s, data, args.epochs, args.lr, args.batch_size, args.patience) for s in args.seeds]

    accs = [r["accuracy"] for r in results]
    aurocs = [r["auroc"] for r in results]
    print()
    print(f"model: {args.model}")
    print(f"val accuracy: mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
    print(f"val AUROC: mean={np.mean(aurocs):.4f} std={np.std(aurocs):.4f}")
    print(f"trainable params: {results[0]['params']}")


if __name__ == "__main__":
    main()
