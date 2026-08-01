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

## Validation split: known simplification

The official split is participant-based (10 training participants), but the UEA archive does not
retain per-trial participant IDs, and recovering them exactly would require re-downloading the
original Kaggle competition source data. Since the goal of this project is architecture practice
rather than a publishable result, we accept this limitation explicitly rather than blocking on it:

- Internal train/validation splitting uses a stratified random split of the official training
  partition (fixed seed, same split reused across models for a fair comparison).
- This almost certainly understates the true generalisation gap, because trials from the same
  participant can land in both the internal train and validation folds. Validation accuracy should
  therefore be read as optimistic relative to the untouched official test set.
- The official test set (held-out participants) is still the only number we trust for the
  headline comparison between architectures, and it is still never used for model or
  hyperparameter selection.
