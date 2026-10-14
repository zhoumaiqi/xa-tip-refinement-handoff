# Phase 10 Data Augmentation Report

## Scope

This phase tested training-time augmentation for the existing local catheter-tip
refiner. The model, loss, crop size, heatmap decoder, alpha, optimizer, and
evaluation protocol were held fixed:

- HRNet-W18-small-v2, ImageNet pretrained
- raw-centered 256 x 256 crop
- 64 x 64 Gaussian heatmap
- weighted MSE and integer argmax decoding
- fixed correction alpha = 0.75
- 8-case leave-one-case-out OOF
- clean, unaugmented OOF inference

Frozen Val and reserved Final Test cases `908` and `771` were not accessed.

## First Screening

| Condition | Case-macro mean (px) | Mean (px) | P95 (px) | Harmful >10 px |
|---|---:|---:|---:|---:|
| A0 none | 16.82 | 15.51 | 50.14 | 3.48% |
| A1 intensity | 17.73 | 16.29 | 48.37 | 4.49% |
| A2 jitter32 | 16.59 | 15.52 | 59.43 | 16.05% |

Interpretation:

- Broad intensity augmentation was not helpful in this short, one-seed
  screening run.
- 32 px crop-center jitter was too aggressive. It created many new severe
  errors even though its mean error was slightly lower.

## Conservative Jitter Sweep

| Condition | Case-macro mean (px) | Mean (px) | P95 (px) | Harmful >10 px |
|---|---:|---:|---:|---:|
| A0 none | 16.82 | 15.51 | 50.14 | 3.48% |
| A2 jitter16 | 15.13 | 14.21 | 48.74 | 9.82% |
| A2 jitter8 | 15.11 | 13.31 | 48.76 | 5.80% |

The 8 px condition was selected for multi-seed confirmation.

## Four-Seed Confirmation

This confirmation used seeds `20260919` through `20260922`, with the same
8-case OOF protocol.

| Condition | Case-macro mean (px) | Mean (px) | P95 (px) | P99 (px) | Harmful >10 px | Raw-good harmful >10 px |
|---|---:|---:|---:|---:|---:|---:|
| A0 none | 17.19 +/- 0.07 | 15.92 +/- 0.09 | 48.70 +/- 0.94 | 56.18 +/- 1.28 | 4.15% +/- 0.88% | 6.26% +/- 1.20% |
| A2 jitter8 | 14.51 +/- 0.28 | 13.58 +/- 0.27 | 47.89 +/- 0.69 | 57.81 +/- 2.19 | 5.14% +/- 0.34% | 5.81% +/- 0.80% |

The jitter8 condition improved the case-macro mean in every confirmation
seed:

| Seed | Delta case-macro mean, jitter8 - none (px) |
|---:|---:|
| 20260919 | -2.98 |
| 20260920 | -2.27 |
| 20260921 | -2.54 |
| 20260922 | -2.93 |

At the individual case/seed level, 84.4% of comparisons improved and 15.6%
became worse. The largest gains came from difficult cases, especially
`202607130347`, while a few easy cases such as `131` became slightly worse.

## Decision

Keep `A2_jitter8` as the current augmentation candidate for one controlled
Frozen Val confirmation. Do not use `jitter16`, `jitter32`, or the current
intensity recipe in the next confirmation.

This is not yet a claim of universal safety. Jitter8 lowers average error and
slightly lowers the `raw-good` harmful rate on average, but its overall
harmful-over-10-px rate is approximately one percentage point higher than the
matched baseline. The Frozen Val confirmation must therefore report both
accuracy and harm metrics.

## Next Step

Freeze the following policy before reading Frozen Val:

1. Train the unchanged model with `A2_jitter8`.
2. Use the existing training/development data only.
3. Read Frozen Val once as confirmation.
4. Report mean, case-macro mean, median, P95/P99, harmful >5/>10 px,
   raw-good harm, and per-case results.
5. Do not tune alpha, decoder, threshold, or model size using that Frozen Val
   result.

If the Frozen Val result is directionally consistent, retain jitter8 as the
augmentation baseline and move to the next research question: confidence-aware
safe non-correction or coordinate-offset prediction. Segmentation should be
introduced later as structural supervision, not mixed into this augmentation
decision.
