# Phase 9A — Extended Dataset Eligibility Audit

## Plain-language answer

- The current Train has 8 independent cases. The fresh technical audit found **6 new cases** with clean CSV↔DICOM mapping that could enter an A+B candidate pool after required human visual review.
- Tier A alone keeps Train at **8 cases / 12 Series / 2784 frames**.
- Tier A+B can conditionally reach at most **14 cases / 22 Series / 5781 labeled frames** while keeping Val frozen; every new Tier B Series remains pending human visual review.
- Among previously excluded annotated Series, **11 Series** are Tier B because CSV/DICOM/coordinates/pixels are clean and the remaining issue is auxiliary JSON absence/count mismatch. Another **3 Series** are technically clean and have JSON mismatch, but remain Tier C because the historical record contains an annotation-quality concern that still requires manual review.
- Explicit Tier D exclusion affects **1 annotated Series**; the principal confirmed issue is the unresolved 6700/XA-03 annotation conflict. Unlabeled XA is reported separately and is not given a fabricated training label.
- Added A+B frames comprise **2997 frames from genuinely new cases** and **0 frames from extra Series of existing Train cases**.
- Frozen Val preserved: **YES**. Cases 13424, 2148, and 6700 are excluded from every expansion scenario.
- With 6 technically eligible new cases, it is possible to expand Train and still reserve at least two new cases for a future final Test; Phase 9A does not choose that split.

**Conclusion 1: existing data support a meaningful expansion of independent Train cases, so an Extended-Train single-frame baseline is worth considering only after human review.**

## Inventory reconciliation

- Fresh scan: 19 cases, 57 XA Series, 12752 XA DICOM frames.
- Primary CSV annotations: 36 Series and 9367 unique physical DICOM frames. The other 21 Series / 3385 frames have no primary CSV and are kept separately as UNLABELED.
- Historical reference: 19 cases, 57 XA Series, 12,752 frames.
- Difference: +0 cases, +0 Series, +0 frames.

## Tier totals

| Tier | Cases | Series | Labeled frames |
|---|---:|---:|---:|
| A | 11 | 16 | 3712 |
| B | 7 | 11 | 3229 |
| C | 4 | 8 | 2195 |
| D | 1 | 1 | 231 |

Tier B is technically eligible but remains **pending human visual review**. Tier C is not automatically trainable. Tier D is excluded. JSON is auxiliary and never the true-tip GT authority.

## New-case candidates

| Case | Eligible Series | Eligible frames | Tier | Main issue | Human review required |
|---|---:|---:|---|---|---|
| 14-cbct | 1 | 232 | B | CSV/DICOM clean; auxiliary JSON missing | yes |
| 2768 | 2 | 463 | B | CSV/DICOM clean; auxiliary JSON count_mismatch | yes |
| 4274 | 1 | 402 | B | CSV/DICOM clean; auxiliary JSON count_mismatch | yes |
| 658 | 1 | 232 | B | CSV/DICOM clean; auxiliary JSON missing | yes |
| 771 | 4 | 1437 | B | CSV/DICOM clean; auxiliary JSON count_mismatch | yes |
| 908 | 1 | 231 | B | CSV/DICOM clean; auxiliary JSON count_mismatch | yes |

## Human visual review checklist

- [ ] raw point looks consistent with the system coarse localization
- [ ] true_tip lies near the intended catheter tip
- [ ] no obvious frame/label shift
- [ ] no row/column swap pattern
- [ ] no sustained label drift

The generated preview images support review but do not prove medical label correctness. No original file was modified, no model was trained or run, no split was changed, and Test was not used.
