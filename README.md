# When Does Attention Help in Multivariate Time-Series Classification?

This is a PyTorch portfolio project that studies a narrow question carefully:

> Under a matched parameter and training budget, does temporal self-attention improve
> cross-subject MEG face-versus-scramble classification relative to convolutional and
> recurrent models?

## Result

Yes, clearly. Five models were implemented from scratch in PyTorch and compared at a matched
~150k-250k trainable-parameter budget:

| Model | Val AUROC |
| --- | --- |
| Pooled linear baseline | 0.578 |
| 1D CNN | 0.657 |
| BiLSTM | 0.660 |
| CNN-Transformer hybrid | 0.794 |
| Transformer encoder | **0.817** |

Self-attention (scaled dot-product, multi-head, sinusoidal positional encoding, pre-norm
encoder blocks) is implemented from scratch in
[src/attention_mts/attention.py](src/attention_mts/attention.py). Local convolution and
recurrence both plateau around the same modest ceiling regardless of mechanism; any model with
direct pairwise attention clears it by a wide margin. Full results, per-model architecture
details, and an attention-weight visualization are in
[docs/results.md](docs/results.md).

## Dataset: UEA FaceDetection

Each example is a 1.5-second MEG recording made while a participant viewed either a face or
a scrambled image. The UEA version contains 5,890 training trials and 3,524 test trials;
every trial has 144 channels and 62 time steps. The official train/test split is by
participant (10 training participants, 6 test participants), so random trial-level
resplitting would produce an invalid evaluation.

The input convention used throughout the project is:

```text
X.shape == (n_examples, n_channels, n_time_steps)
```

## Planned comparison

| Model | Core inductive bias |
| --- | --- |
| Pooled linear baseline | Global channel summary without temporal modelling |
| 1D CNN | Local temporal motifs |
| BiLSTM | Ordered temporal dynamics |
| Transformer encoder | Direct interactions among time points |
| CNN-Transformer hybrid | Local features followed by global attention |

All learned models are matched for parameter count (~150k-250k trainable parameters), optimiser,
and training seeds. Early stopping (best-checkpoint, `patience=10`) is implemented and used for
the Transformer and hybrid; the CNN and BiLSTM were not re-run with it (see
[docs/results.md](docs/results.md) for why) and remain final-epoch numbers.

## Evaluation rules

1. The official test partition is never used for model or hyperparameter selection, and has not
   been touched at all so far — every result to date is on the internal validation split.
2. Validation within the training partition uses a stratified random split, **not** a
   participant-grouped one. The official split is participant-based, but the archive doesn't
   retain per-trial participant IDs; recovering them would require the original Kaggle
   competition source. This is a deliberate, documented simplification for a practice project,
   not an oversight — see
   [experiment-plan.md](docs/experiment-plan.md#validation-split-known-simplification).
3. Channel standardisation is fitted on the relevant training fold only.
4. Results are reported across multiple seeds with accuracy and AUROC (`src/attention_mts/metrics.py`
   also computes balanced accuracy and macro-F1, and can measure inference latency, but only
   accuracy/AUROC are currently surfaced in [docs/results.md](docs/results.md)).

See [docs/experiment-plan.md](docs/experiment-plan.md) for the full research protocol and
[docs/results.md](docs/results.md) for the actual results and their interpretation.

## Setup and first inspection

Install Python 3.10+ and create a virtual environment. This project lives inside a deeply
nested OneDrive path, and `torch`'s installed files include long third-party license paths
that exceed Windows' 260-character `MAX_PATH` when combined with an in-project `.venv/` here
— so the virtual environment is created outside OneDrive instead:

```powershell
py -m venv C:\Users\petar\venvs\attention-mts
C:\Users\petar\venvs\attention-mts\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m attention_mts.inspect --download
```

The inspection command downloads the public dataset to `data/raw/FaceDetection`, prints
tensor dimensions, label distributions, finite-value checks, and per-channel summaries. It
does not shuffle, normalise, or train on the data.

The first completed audit is recorded in [docs/data-audit.md](docs/data-audit.md).

When we begin neural-model implementation, install the training dependency explicitly:

```powershell
python -m pip install -e ".[train,dev]"
```

## What this project covers

- **Data audit** — downloaded, inspected, and validated the public archive before writing any
  model code. See [docs/data-audit.md](docs/data-audit.md).
- **A shared training framework** — deterministic split, per-fold standardisation, shared
  metrics, and a single generalized `scripts/train.py` (best-checkpoint early stopping, any
  registered model) that all five architectures below train through.
- **Five models, matched and compared** — a pooled-linear baseline, 1D CNN, BiLSTM, Transformer
  encoder, and CNN-Transformer hybrid, each solved for a comparable parameter budget and
  compared under the same split, optimiser, and seeds. Self-attention is implemented from
  scratch; the CNN and LSTM use PyTorch's built-in layers, since they were the reference
  comparisons, not the object of study. Full results and interpretation:
  [docs/results.md](docs/results.md).
- **An attention-weight visualization** confirming what the trained Transformer actually
  attends to over time, not just its accuracy — see the same results doc.

**Natural next steps, not pursued here:** the robustness/ablation suite specified in
[docs/experiment-plan.md](docs/experiment-plan.md) (input noise, channel dropout, positional-encoding
and head-count ablations) and evaluation on the held-out test set. The project's scope was set
at answering the headline question above with a clean, well-diagnosed comparison — which it
does — rather than a full research-grade study.
