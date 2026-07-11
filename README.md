# When Does Attention Help in Multivariate Time-Series Classification?

This is a PyTorch portfolio project that studies a narrow question carefully:

> Under a matched parameter and training budget, does temporal self-attention improve
> cross-subject MEG face-versus-scramble classification relative to convolutional and
> recurrent models?

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

All learned models will be matched as closely as practical for parameter count, optimiser,
learning-rate schedule, augmentation, early-stopping rule, and training seeds.

## Evaluation rules

1. The official test partition is never used for model or hyperparameter selection.
2. Validation within the training partition must respect participant groups.
3. Channel standardisation is fitted on the relevant training fold only.
4. Results are reported across multiple seeds with accuracy, balanced accuracy, macro-F1,
   AUROC, parameter count, and inference cost.

See [docs/experiment-plan.md](docs/experiment-plan.md) for the full research protocol.

## Setup and first inspection

Install Python 3.10+ and create a virtual environment, then install the project:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
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

## Project milestones

1. **Data audit** — download, inspect, and verify participant-aware validation metadata.
2. **Shared training framework** — deterministic splits, preprocessing, metrics, and logging.
3. **Controlled models** — baseline, CNN, BiLSTM, Transformer, and hybrid.
4. **Robustness and ablations** — noise, masking, channel dropout, shifts, and architecture tests.
5. **Portfolio report** — mathematics, experimental evidence, visualisations, and limitations.
