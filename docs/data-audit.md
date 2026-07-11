# FaceDetection data audit

Audit run: 2026-07-11  
Loader: `aeon.datasets.load_classification` 1.5.0

## Observed archive contents

| Partition | Tensor shape `(examples, channels, time)` | Label 0 | Label 1 |
| --- | ---: | ---: | ---: |
| Train | `(5890, 144, 62)` | 2,945 | 2,945 |
| Test | `(3524, 144, 62)` | 1,762 | 1,762 |

Both partitions are equal-length, multivariate, and contain only finite values. The loader
metadata reports no timestamps and no missing values. Each channel is already approximately
zero-mean and unit-scale when aggregated over a partition (standard deviation about 0.999).

## Interpretation

The tensor is already in the project convention: each trial has 144 sensor channels observed
at 62 time steps. The exact class balance means plain accuracy and balanced accuracy will have
the same baseline on this fixed split, although we will report both for completeness.

The archive-level scaling does **not** remove the need for a correct preprocessing protocol:
all transformations used in modelling must still be fitted only on the appropriate training
fold. The audit did not alter any data values.

## Participant split: unresolved metadata

The UEA dataset description says the official split holds out participants, but the `aeon`
metadata contains no participant IDs. This means we cannot yet construct a defensible
participant-grouped validation split from the downloaded archive alone.

Before model training, the next data task is to obtain and validate participant identifiers
from the original FaceDetection source, or a documented mapping to archive row ranges. We will
not use a random trial-level validation split as a substitute.
