# Experiment plan

## Scientific claim under test

Attention is not assumed to be better merely because it is more flexible. We test whether
it provides a measurable, robust advantage after controlling for capacity and training
budget on a subject-held-out multivariate classification task.

## Fixed protocol

- **Task:** Face versus scramble classification from 144-channel MEG trials.
- **Official test set:** untouched until the final configuration is frozen.
- **Model selection:** grouped participant-aware validation using only official training data.
- **Preprocessing:** compute a mean and standard deviation for each channel from each training
  fold, then apply those values to that fold's validation data and the final test data.
- **Repetition:** at least five random initialisation seeds per final configuration.

## Model budget

The initial target is approximately 150k--250k trainable parameters per learned model. The
exact width/depth values will be solved from parameter counts before training rather than
chosen independently per architecture. Every model receives the same `(channels, time)` input.

## Robustness suite

- additive sensor noise;
- random channel dropout;
- contiguous temporal masking;
- small temporal shifts.

Each perturbation is evaluated at several strengths. We will report performance degradation,
not only clean-test accuracy.

## Required ablations

1. Transformer positional encoding and pooling method.
2. Number of Transformer blocks and attention heads at near-matched capacity.
3. Hybrid convolutional stem versus direct tokenisation.
4. Temporal downsampling factor in the hybrid.

## Important validity gate

The dataset description states that the official split is participant-based. Before modelling,
we must establish reliable participant identifiers or a defensible reconstruction of training
participant groups. If the archive files do not retain this information, we will obtain it from
the original data source or revise the validation design; we will not silently use random
trial-level validation.

The project code enforces this policy: grouped-validation utilities require an explicit,
one-identifier-per-trial group array and raise an error when it is absent.
