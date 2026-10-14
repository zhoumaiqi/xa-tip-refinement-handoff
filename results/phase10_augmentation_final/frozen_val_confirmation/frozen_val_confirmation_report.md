# Phase 10 Frozen Val Confirmation

## Protocol

This was the single Frozen Val confirmation planned after OOF selection.

- Frozen Val frames: 928
- Frozen Val cases: 3 (`2148`, `6700`, `13424`)
- Final Test cases `908` and `771`: not used
- Candidate: saved all-development-train `A2_jitter8` checkpoint
- Baseline: matched all-development-train `A0_none` model, same seed and
  training schedule
- Both models used clean, unaugmented inference on Frozen Val.

## Results

| Condition | Mean (px) | Case-macro mean (px) | Median (px) | P95 (px) | P99 (px) | Harmful >10 px | Raw-good harmful >10 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 none | 22.67 | 19.02 | 15.61 | 49.90 | 52.62 | 5.50% | 10.04% |
| A2 jitter8 | 20.39 | 18.14 | 13.97 | 53.84 | 59.86 | 18.00% | 26.16% |

## Interpretation

`A2_jitter8` improved average localization metrics on Frozen Val:

- mean error improved by 2.28 px;
- case-macro mean improved by 0.88 px;
- median error improved by 1.65 px.

However, it substantially worsened safety and tail metrics:

- P95 increased by 3.94 px;
- harmful >10 px increased from 5.50% to 18.00%;
- raw-good harmful >10 px increased from 10.04% to 26.16%.

The improvement was concentrated in difficult case `13424`. Cases `2148` and
`6700` became worse on mean error, and `6700` had especially high raw-good
harm under jitter8.

## Decision

The saved `A2_jitter8` checkpoint is technically valid and should be retained
as a research artifact. It should not be promoted as the default safe model
based on this one Frozen Val confirmation, because its severe-harm metrics are
materially worse than the matched A0 baseline.

Frozen Val has now been read once for this augmentation decision. Do not tune
jitter range, alpha, decoder, or other settings against this same Frozen Val
result. Future work should use the existing OOF predictions to design a
confidence-aware non-correction policy, then evaluate that policy with a new
predefined protocol.
