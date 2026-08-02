# Results

Validation metrics are computed on the stratified internal split (`SPLIT_SEED = 0`, 20% of the
official training partition — see [experiment-plan.md](experiment-plan.md#validation-split-known-simplification)
for why this is a simplification, not a participant-grouped split). Each row is mean ± std over
5 initialisation seeds (`--seeds 0 1 2 3 4`), same split reused across all seeds and models.
The official test set has not been touched.

| Model | Trainable params | Val accuracy | Val AUROC |
| --- | ---: | --- | --- |
| Pooled linear baseline | 145 | 0.559 ± 0.003 | 0.578 ± 0.002 |
| 1D CNN (uncorrected, see caveat) | 169,201 | 0.589 ± 0.020 | 0.657 ± 0.010 |
| BiLSTM (uncorrected, see caveat) | 186,049 | 0.610 ± 0.011 | 0.660 ± 0.006 |
| CNN-Transformer hybrid (uncorrected, see caveat) | 176,289 | 0.713 ± 0.010 | 0.791 ± 0.009 |
| Transformer encoder (uncorrected, see caveat) | 163,777 | 0.731 ± 0.004 | 0.808 ± 0.006 |

## Transformer encoder

Command: `python scripts/train.py --model transformer`

Architecture: `d_model=96, n_heads=4, d_ff=192, n_layers=2, dropout=0.1`, pre-norm blocks,
sinusoidal positional encoding, mean pooling over time (see `src/attention_mts/attention.py`
and the `TransformerClassifier` class in `src/attention_mts/models.py`).

**Same known caveat as the CNN — final-epoch, not best-checkpoint, result.** All 5 seeds
overfit: train loss collapses toward ~0.01-0.05 by epoch 30 while val loss bottoms out around
epoch 3-5 (e.g. seed 0's minimum is 0.516 at epoch 4) and then climbs to 1.3-1.6+ by epoch 30.

One nuance that makes this caveat less damaging than it sounds: unlike a simple picture of
overfitting, val accuracy/AUROC stayed high even as val loss climbed. Binary cross-entropy
punishes *confident* wrong predictions heavily, so a rising loss with a stable/high accuracy is
consistent with the model becoming more overconfident on both its correct and incorrect
predictions, rather than actually flipping more predictions to wrong. So the true early-stopped
accuracy could be similar, or could differ in either direction — this hasn't been checked
directly. What's not in question: this final-epoch number is already a wide, unambiguous margin
over both other uncorrected models (73.1% vs. 58.9% CNN vs. 55.9% baseline; AUROC 0.808 vs.
0.657 vs. 0.578), so the qualitative conclusion — attention provides a real, structural
advantage on this task at matched parameter budget — holds regardless of exactly which epoch is
reported.

## CNN-Transformer hybrid

Command: `python scripts/train.py --model hybrid`

Architecture: `Conv1d` channel projection (`n_mid=32`) into a stride-2, `kernel_size=7` conv
(`d_model=96`) that downsamples 62 time steps to 31, then `n_heads=4, d_ff=192, n_layers=2`
Transformer encoder blocks over the downsampled sequence (see `CNNTransformerHybrid` in
`src/attention_mts/models.py`).

**Same known caveat as the other learned models — final-epoch, not best-checkpoint, result.**
All 5 seeds overfit similarly: train loss falls toward ~0.01-0.07 by epoch 30 while val loss
bottoms out around epoch 3-4 (e.g. seed 1's minimum is 0.545 at epoch 4) then climbs to
1.4-1.7+ by epoch 30.

This result directly answers one of the two required ablations from `experiment-plan.md`
("hybrid convolutional stem vs. direct tokenisation"): the hybrid (71.3% / 0.791) comes close
to but does not quite match the plain Transformer (73.1% / 0.808), despite attention here
operating on only 31 downsampled tokens instead of 62 raw ones. So compressing time before
attention costs a small amount of accuracy — the conv stem's downsampling discards some
information direct attention over all 62 steps could use, and that outweighs the benefit of
giving attention fewer, pre-aggregated tokens to work with, at least at `downsample_factor=2`
and this parameter budget. At the same time, the hybrid still clearly beats the CNN (58.9%) and
BiLSTM (61.0%) by a wide margin, so most of the benefit in this study is coming from attention
existing at all, not from the specific way tokens are constructed before it.

## Attention visualization

Command: `python scripts/visualize_attention.py`

![Block 0 attention weights, 4 heads, one correctly-classified example per class](attention_block0_heads.png)

A separate, lightly-trained (5 epochs, seed 0, near the val-loss-minimum region seen in the
full run) model, used only for interpretability — `train.py` does not checkpoint weights, so
this is not the same model instance as the reported 30-epoch result. Shows block 0's 4 attention
heads for one confidently-correct "face" and one confidently-correct "scramble" validation
trial. The dashed line marks the stimulus-onset time step (step 21 of 62, from the dataset's
documented 0.5s pre-stimulus baseline within a 1.5s trial).

**Dominant pattern: vertical stripes ("attention sinks"), not diagonals.** Nearly every query
time step attends most strongly to the same handful of *key* time steps, rather than to keys
near itself — the opposite of what a "local window" pattern would look like. This suggests the
model routes most of its information through a few landmark positions whose value vectors carry
broadly useful summary content, rather than distributing attention based on temporal proximity.

**Only partial support for the M170 hypothesis.** Some heads' landmark columns land right around
or just after stimulus onset (head 0 near step ~26-28, one of head 2's columns near step
~20-30) — consistent with the model anchoring on the post-stimulus evoked-response window
predicted earlier. But head 1's landmark sits just *before* onset (~15-18), and head 3 has
strong columns much later (~40-58), outside the M170 timeframe. So this is not a single clean
"the model discovered the 150ms evoked response" result — different heads appear to use
different landmark strategies, only some of which match the neuroscience-motivated hypothesis.

**Not yet established: whether these landmarks are stable, class-conditioned structure.** This
is one model, one seed, two example trials. Confirming these landmark positions reflect genuine
task structure (rather than this particular model/example's quirk) would need aggregating
attention patterns across many validation examples and seeds — not yet done.

## BiLSTM

Command: `python scripts/train.py --model bilstm`

Architecture: single-layer bidirectional `nn.LSTM`, `hidden_size=96` (so 192-dim output per
time step after concatenating both directions), mean pooling over time, linear head (see
`BiLSTMBaseline` in `src/attention_mts/models.py`).

**Same known caveat as the CNN and Transformer — final-epoch, not best-checkpoint, result.**
All 5 seeds overfit: train loss collapses toward ~0 by epoch 30 (e.g. seed 4: 0.0013) while val
loss bottoms out around epoch 3-4 (e.g. seed 4's minimum is 0.659 at epoch 4) and then climbs to
1.4-2.4+ by epoch 30.

The result itself is more informative than the caveat this time. The BiLSTM's val AUROC (0.660)
is essentially tied with the CNN's (0.657) and far behind the Transformer's (0.808), despite the
BiLSTM having a mechanism the Transformer had to fake: order is implicit in the recurrence
itself, no positional encoding needed, and in principle a recurrent model can propagate
information across arbitrarily long ranges. That it doesn't outperform a model with a fixed,
short local receptive field (the CNN, receptive field ~19 of 62 steps) is a sharper piece of
evidence than the CNN comparison alone for *why* attention helps here specifically: the
bottleneck isn't "having access to order," it's that an LSTM can only move information from time
step 5 to time step 50 by successfully carrying it through 45 sequential updates, while
self-attention connects any two time steps directly in a single operation. Since all three
learned models sit in a similar parameter range (164k-186k), this isn't a capacity story either
— it looks like a genuine structural advantage of direct pairwise attention over both convolution
and recurrence on this task.

## 1D CNN

Command: `python scripts/train.py --model cnn`

**Known caveat — not an early-stopped/best-checkpoint result.** The training loop reports
metrics from whatever the final epoch (30) produced, not the best epoch. All 5 seeds show
severe overfitting: train loss collapses to near-zero (e.g. seed 0: 0.028, seed 4: 0.020) while
val loss bottoms out around epoch 4-7 (e.g. seed 0's minimum is 0.678 at epoch 4) and then
climbs steadily to 1.5-3.0+ by epoch 30. The reported numbers above are therefore measuring an
overfit model, not this architecture's best achievable validation performance — the true number
is likely somewhat higher. Fixing this requires adding best-checkpoint early stopping to
`scripts/train.py` (tracked as follow-up work, not yet implemented — deprioritised in favour of
moving on to the Transformer, which is the actual point of this project).

Even with that caveat, the qualitative comparison to the baseline is informative: within just 2
epochs the CNN already matched the baseline's fully-converged accuracy, and its AUROC (0.657)
clears the baseline's ceiling (0.578) by a wide margin despite the uncorrected number being an
underestimate. This is consistent with the mean-pooling interpretation below — a model that can
localise a brief motif anywhere in the 62 time steps starts from a real structural advantage
over one that can only see a time-averaged summary.

## Pooled linear baseline

Command: `python scripts/train_baseline.py` (superseded by `python scripts/train.py --model baseline`)

Mean-pooling each channel over the full 1.5 s window and fitting a linear classifier on the
result barely beats chance (accuracy 55.9%, AUROC 0.578). A longer diagnostic run (300 epochs,
lr=1e-2, single seed) confirms this is a genuine ceiling rather than undertraining: even
**train** accuracy only reaches 59.4% (AUROC 0.637) on a convex problem with a guaranteed
global optimum, and val loss starts rising slightly past ~150 epochs while train loss keeps
falling — mild overfitting on top of an already-low ceiling.

Interpretation: face-vs-scramble MEG evoked responses are transient, oscillatory deflections
(e.g. an M170-type component lasting on the order of 100ms within the 1.5s trial). Averaging
over all 62 time steps lets such a brief deflection cancel toward the trial's baseline level,
so the pooled representation discards most of the class-relevant signal before the classifier
ever sees it. This is a stronger and more specific reason to expect temporal models (CNN,
BiLSTM, Transformer) to outperform this baseline than "more parameters generally help" — they
have a concrete mechanism (a receptive field, recurrence, or attention) for keeping the
information this baseline throws away.