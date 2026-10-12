# XA Tip Refinement — READ THIS FIRST

## What this package is

This is the final engineering and research handoff for the XA catheter-tip refinement project through Phase 9C (2026-09-24). It contains the runnable source, frozen configs and split records, the current baseline checkpoint, five traceable Extended-12 candidate checkpoints, compact experiment evidence, and integrity manifests. It deliberately contains **no raw DICOM pixels**.

The product task is **coarse-to-fine catheter-tip refinement**:

1. The product supplies a coarse tip point `raw_col/raw_row` on the current XA frame.
2. A 256×256 local crop is centered on that deployable coarse point.
3. HRNet predicts a local heatmap candidate.
4. A fixed fraction of the candidate correction is applied: `final = raw + alpha × (candidate - raw)`.

`raw_row/raw_col` are the product's automatic coarse coordinates in the original image coordinate system. `true_tip_row/true_tip_col` are manually corrected final ground truth and are supervision/evaluation only; they must never be used as the crop center or inference input.

## Current recommended baseline

The most reliable baseline remains the **Phase 6G-style single-frame local refiner**:

- current XA frame only, grayscale replicated to three channels;
- raw-centered 256×256 crop;
- ImageNet-pretrained `hrnet_w18_small_v2`;
- 64×64 one-channel heatmap, Gaussian sigma 2;
- weighted MSE `mean((1 + 9×GT) × (prediction-GT)^2)`;
- integer heatmap argmax mapped back to the original image;
- fixed historical `alpha = 0.75`.

Phase 9B's Extended-12 five-seed models and Phase 9C's `alpha=0.50` are included as **safety-oriented research candidates**, not as a replacement main baseline. Extended-12 did not improve the primary matched old-case OOF mean reliably (1/5 seeds), although severe and persistent harmful corrections decreased. Alpha 0.50 improved OOF safety/tail behavior but did not improve frozen-Val mean.

## Project status

The end-to-end system-to-local automatic refinement pipeline is validated. It can improve many coarse locations and has demonstrated reductions in severe regression and persistent harmful errors. Cross-case hard-case refinement remains unstable, however, and the system is **not ready to replace manual calibration**.

The main remaining weaknesses are cross-case distribution differences, unnecessary correction of already-good raw points, inconsistent hard-case rescue, persistent wrong-track/wrong-structure errors, and heterogeneous correction-strength preference across cases. Evidence does **not** indicate that the primary blocker is the coordinate convention, CSV/DICOM alignment, 256 ROI coverage, heatmap resolution, or merely the choice of alpha.

## Frozen data partitions

| Partition | Cases | Series | Frames | Status |
|---|---:|---:|---:|---|
| Original Train | 8 | 12 | 2,784 | historical training |
| Extended new Train | 4 (`4274`, `14-cbct`, `2768`, `658`) | 5 | 1,329 | human-reviewed |
| Extended Train total | 12 | 17 | 4,113 | Phase 9B candidate training |
| Frozen Val | 3 (`13424`, `2148`, `6700`) | 4 | 928 | repeatedly observed validation |
| Reserved Final Test | 2 (`908`, `771`) | 5 | 1,668 | **SEALED / NOT USED** |

The deterministic split hash is `db0944eb9c26ce0a310edae029fff9dcfb507cac7bf4cac9c127dfbbff85844e`.

## CRITICAL: Final Test must not be touched

Cases **908 and 771** are the reserved Final Test. Through Phase 9C they have had zero pixel reads, GT reads, model inference, or metric calculations; `test_used=false`. Do not inspect images, labels, predictions, or metrics from these cases until the complete final model policy is frozen: training cases, architecture, preprocessing, seed/ensemble rule, epochs, alpha, checkpoint-selection rule, and reporting plan. Read [FINAL_TEST_DO_NOT_TOUCH.md](FINAL_TEST_DO_NOT_TOUCH.md) before any future evaluation.

## What was learned

- Full-frame localization learned position shortcuts and wrong structures; its cross-case tail was unacceptable.
- Product `raw_*` semantics were confirmed. A raw-centered 256 ROI covers the observed single- and three-frame geometry, so crop geometry is not the current bottleneck.
- Phase 6G is the dependable engineering anchor, but it can over-correct easy points and miss difficult cases.
- Direct causal three-frame channel stacking produced a repeatable OOF signal (5/5 seeds, about 2 px mean improvement) but won only 1/5 seeds on independent Val. Temporal information exists, but the representation did not generalize reliably.
- Shared-backbone late feature fusion failed its Stage-1 gate; explicit temporal differences showed some OOF benefit but 0/5 Val wins.
- Repeat-seed experiments showed ordinary training variance but also showed that the temporal OOF signal was not a single-seed accident.
- Six new independent cases (10 Series, 2,997 annotated frames) passed technical and human visual review. Four were assigned to Extended Train and two were sealed as Final Test by deterministic hash.
- More cases improved safety but not the primary matched OOF mean reliably. Recalibrating alpha to 0.50 further improved safety/tail trade-offs, not generalization mean.

## Recommended next action

Before opening Final Test, do a human failure-phenotype review of persistent wrong-track and hard-rescue failures using only Train/Val artifacts. If model work continues, prioritize additional independent cases and explicit, alignment-aware temporal research; do not simply add 5/7-frame channel stacks. The reserved Test should be used only once after a single final policy is formally frozen.

Start with [01_STAGE_SUMMARY.md](01_STAGE_SUMMARY.md), then [02_HOW_TO_RUN.md](02_HOW_TO_RUN.md) and [03_DATA_AND_SPLIT.md](03_DATA_AND_SPLIT.md).
