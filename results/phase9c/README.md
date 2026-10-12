# Phase 9C — Fixed Refinement-Strength Recalibration Audit

## Plain-language result

- BASELINE-8 OOF-selected global alpha: **0.50** (historical anchor 0.75).
- EXTENDED-12 OOF-selected global alpha: **0.50**.
- EXTENDED OOF case-macro Mean: alpha .75 = **17.409px**; selected alpha = **17.072px**.
- Strict nested EXTENDED: selected = **17.244px** vs fixed .75 = **17.409px**; outcomes 19 improved / 16 worsened / 5 ties out of 40.
- Frozen Val EXTENDED: alpha .75 Mean = **21.257px**; selected alpha Mean = **21.315px**.
- Final choice: **2** — Recalibration changes the safety/rescue trade-off but does not solve generalization.
- No training, model inference, checkpoint loading, or Final Test access occurred.

The practical direction is **more conservative correction**: alpha 0.50 accepts half of the raw-to-candidate displacement, rather than 75%.

## Candidate and isolation audit

- All four saved tables (BASELINE/EXTENDED × OOF/Val) contain direct `candidate_x/y` coordinates.
- Reapplying alpha 0.75 reproduced the saved historical final coordinates with maximum discrepancy below `8.6e-14 px`; no candidate reconstruction or model inference was needed.
- OOF contains the same 2760 old-case frames for both methods and five seeds (13,800 rows/method).
- Val contains the same 920 frozen frames for both methods and five seeds (4,600 rows/method).
- Reserved cases `908` and `771`: pixel reads 0, GT reads 0, inference reads 0, metric reads 0; `test_used=false`.

## Nested alpha choices (40 outer case-seed evaluations per method)

| alpha | BASELINE-8 | EXTENDED-12 |
|---:|---:|---:|
| 0.00 | 0 | 0 |
| 0.25 | 2 | 8 |
| 0.50 | 28 | 28 |
| 0.75 | 10 | 4 |
| 1.00 | 0 | 0 |

## OOF global alpha landscape

The selection metric is the five-seed mean of the eight-case macro Mean; ties within 0.05px choose the smaller alpha.

| Method | alpha | Case-macro Mean | P95 | Harmful >10 |
|---|---:|---:|---:|---:|
| BASELINE_8 | 0.00 | 18.056 | 50.396 | 0.00% |
| BASELINE_8 | 0.25 | 17.323 | 49.033 | 1.28% |
| BASELINE_8 | 0.50 | 16.973 | 49.615 | 3.03% |
| BASELINE_8 | 0.75 | 17.044 | 50.671 | 6.40% |
| BASELINE_8 | 1.00 | 17.695 | 53.412 | 11.30% |
| EXTENDED_12 | 0.00 | 18.056 | 50.396 | 0.00% |
| EXTENDED_12 | 0.25 | 17.262 | 48.671 | 0.53% |
| EXTENDED_12 | 0.50 | 17.072 | 48.722 | 1.94% |
| EXTENDED_12 | 0.75 | 17.409 | 49.731 | 4.11% |
| EXTENDED_12 | 1.00 | 18.255 | 50.598 | 11.59% |

For BASELINE-8, alpha 0.50 improves the all-case OOF macro Mean only slightly (17.044 → 16.973 px), while strict nested selection is worse than fixed 0.75 (17.318 vs 17.044 px). Thus 0.75 remains a defensible near-optimal historical anchor for BASELINE-8; recalibration is not cross-case robust there.

For EXTENDED-12, alpha 0.50 improves all-case OOF macro Mean from 17.409 → 17.072 px. In the strict nested audit it improves 17.409 → 17.244 px, with 19 improved / 16 worsened / 5 tied outer case-seed evaluations. The nested choice is concentrated at 0.50 (28/40), but the margin over 0.75 is modest rather than universal.

## Case alpha heterogeneity

EXTENDED-12 five-seed per-case preferences span the full grid:

| Old case | Best alpha |
|---|---:|
| 131 | 0.75 |
| 15716 | 1.00 |
| 20260706004 | 0.25 |
| 202607130347 | 0.25 |
| 361 | 0.50 |
| 4904 | 0.75 |
| 6702 | 0.00 |
| 9950 | 0.00 |

Therefore a global alpha has a real capacity limit even though 0.50 is the best overall compromise.

## Easy safety and hard rescue

For EXTENDED-12 OOF raw <5 px frames, alpha 0.50 versus 0.75 changes mean error 5.752 → 3.996 px and harmful >10 from 7.47% → 4.59%. On Val, the same comparison changes mean error 8.204 → 5.951 px and harmful >10 from 11.74% → 4.51%.

Hard-case rescue is not meaningfully restored. OOF raw 20–40 px mean gain changes 1.514 → 1.600 px, while raw 40–60 changes 1.419 → 1.302 px. Val rescue decreases in both bins: 9.229 → 6.430 px for raw 20–40, and 1.457 → 1.297 px for raw 40–60.

## Frozen Val confirmation

EXTENDED-12 alpha 0.50 versus historical 0.75:

| Metric (5-seed mean) | alpha 0.75 | alpha 0.50 |
|---|---:|---:|
| Mean | **21.257** | 21.315 |
| Across-seed Mean SD | 0.590 | **0.483** |
| Median | **14.480** | 16.797 |
| P95 | 47.853 | **46.214** |
| P99 | 53.216 | **49.831** |
| Harmful >5 | 25.98% | **14.15%** |
| Harmful >10 | 7.04% | **2.15%** |
| Harmful >20 | 1.41% | **0.28%** |

Persistent harmful frames average 35.0 → 8.4 and maximum harmful segment 9.0 → 4.4 frames. However, the primary Val Mean does not improve.

By Val case (Mean error):

| Case | alpha 0.75 | alpha 0.50 | Change |
|---|---:|---:|---:|
| 13424 | 31.126 | 32.783 | +1.657 (worse) |
| 2148 | 14.260 | 13.076 | -1.184 (better) |
| 6700 | 8.517 | 6.619 | -1.897 (better) |

Alpha 0.50 further protects the easy `6700` case and improves `2148`, but loses rescue on difficult `13424`. This is a safety/rescue redistribution, not a general solution.

## Interpretation limits

Alpha recalibration is deterministic post-processing of saved raw/candidate coordinates. It cannot establish that data expansion improved the visual model itself. Nested selection is an audit of cross-case stability, not a case-adaptive deployment rule. Final Test cases 908 and 771 remain sealed.
