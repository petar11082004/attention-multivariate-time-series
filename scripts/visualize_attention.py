"""Visualize what the trained Transformer's attention heads actually attend to over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from attention_mts.models import TransformerClassifier
from train import prepare_data, set_seed

SEED = 0
EPOCHS = 5  # near the best val_loss epoch seen in the full run, before heavy overfitting sets in

# 1.5s trial, 62 time steps, stimulus onset at 0.5s in (see README dataset description)
STIMULUS_STEP = round(0.5 / 1.5 * 62)


def train_for_visualization(data) -> TransformerClassifier:
    set_seed(SEED)
    (X_train_t, y_train_t), _ = data
    model = TransformerClassifier(
        n_channels = X_train_t.shape[1], d_model=96, n_heads=4, d_ff=192, n_layers=2, dropout = 0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_train = X_train_t.shape[0]
    for _ in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n_train)
        for start in range(0, n_train, 128):
            idx = permutation[start: start + 128]
            optimizer.zero_grad()
            logits = model(X_train_t[idx]).squeeze(-1)
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
    return model


def find_confident_correct_example(model, X_val_t, y_val_t, class_label: int) -> int:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t).squeeze(-1))
    preds = (probs >= 0.5).long()
    correct = preds == y_val_t.long()
    mask = (y_val_t.long() == class_label) & correct
    candidates = mask.nonzero(as_tuple=True)[0]
    return candidates[0].item()


def main() -> None:
    data = prepare_data()
    (_, _), (X_val_t, y_val_t) = data
    model = train_for_visualization(data)
    model.eval()
    
    class_names = {0: "scramble", 1: "face"}
    example_indices = {c: find_confident_correct_example(model, X_val_t, y_val_t, c) for c in (0,1)}

    fig, axes = plt.subplots(2 ,4 , figsize = (16, 8))
    for row, (class_label, idx) in enumerate(example_indices.items()):
        with torch.no_grad():
            model(X_val_t[idx: idx + 1])
            attn = model.blocks[0].attention.last_attention_weights[0] # (heads, seq, seq)

            for head in range(attn.shape[0]):
                ax = axes[row, head]
                im = ax.imshow(attn[head].numpy(), cmap = "viridis", aspect = "auto")
                ax.axvline(STIMULUS_STEP, color="white", linestyle="--", linewidth=1)
                ax.axhline(STIMULUS_STEP, color="white", linestyle="--", linewidth=1)
                ax.set_title(f"{class_names[class_label]}, head {head}")
                if head == 0:
                    ax.set_ylabel("query time step")
                if row == 1:
                    ax.set_xlabel("key time step")

    fig.colorbar(im, ax=axes, shrink=0.6, label="attention weight")
    out_path = "docs/attention_block0_heads.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"stimulus onset at time step {STIMULUS_STEP} (dashed white line)")


if __name__ == "__main__":
    main()