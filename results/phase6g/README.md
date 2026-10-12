# Phase 6G — Case-OOF Conservative Heatmap Shrinkage

## Status and frozen design

Completed. The prescribed 8-case leave-one-case-out experiment used Phase 6B's raw-centered 256×256 HRNet heatmap model, fresh ImageNet initialization per fold, weighted MSE, and no augmentation/jitter/noise/temporal input. Every OOF prediction came from a model that did not see that case. No Test data were used.

The alpha grid was fixed in advance: `0, 0.25, 0.50, 0.75, 1.00`. **Alpha was selected only by the 8-case OOF case-macro mean error before final Val inference.** The frozen alpha is **0.75**. Final full-Train duration was frozen at the rounded median fold-best epoch: **3 epochs**. Val was run once and was not used to revise alpha or training duration.

## OOF selection

| alpha | case-macro mean px | frame mean px | median px | P95 px | harmful >10 % |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 18.020 | 16.011 | 11.574 | 50.402 | 0.00 |
| 0.25 | 16.763 | 15.070 | 10.361 | 48.655 | 1.54 |
| 0.50 | 15.855 | 14.429 | 9.490 | 49.296 | 2.41 |
| 0.75 | 15.411 | 14.201 | 8.588 | 50.146 | 4.17 |
| 1.00 | 15.799 | 14.666 | 8.322 | 53.260 | 7.11 |


Alpha 0.75 minimized the sole primary criterion (case-macro mean error: 15.411px versus 15.799px for full correction). It also reduced OOF harmful >10px corrections from 7.11% to 4.17%, while retaining most hard-case correction capacity. Case-specific optima were heterogeneous: `{0.0: 2, 0.75: 4, 1.0: 2}` (see `oof_alpha_by_case.csv`), so the global shrinkage is a useful compromise, not a universal case optimum.

## One-shot Val result

| method | Mean px | Median px | P95 px |
|---|---:|---:|---:|
| Raw | 22.299 | 17.527 | 45.319 |
| Phase 6B full correction | 20.008 | 13.139 | 51.908 |
| Phase 6F residual regression | 22.865 | 21.056 | 46.006 |
| Phase 6G alpha=0.75 | 20.003 | 13.753 | 48.976 |

Phase 6G achieved the lowest overall Val mean (20.003px), essentially tied with 6B on mean but with a lower P95 (48.976 vs 51.908px) and substantially fewer harmful >10px corrections (8.51% vs 15.95%). It does not make easy cases fully safe: for raw-error 0–5px, 6G still worsened most frames and remains materially less safe than 6F.

## Interpretation

The evidence supports outcome **2**: fixed shrinkage helps the safety/rescue tradeoff, but held-out cases prefer materially different alpha values. Phase 6G is worth retaining as a stronger fixed-fusion baseline, while any future adaptive fusion must be calibrated with case-level OOF predictions. No adaptive gate, temporal model, structural supervision, 384 ROI, or Test evaluation was run here.

## Run/recovery audit

The original reporting process terminated after the one-shot Val predictions were safely saved because an Intel OpenMP runtime was initialized twice while starting plotting. Reporting was recovered strictly from already saved OOF and one-shot Val CSV files; training, alpha selection, and Val inference were not rerun. An earlier fold-2 inference indexing failure was repaired by reindexing the held-out `Subset` locally; its saved best fold checkpoint was used without retraining, and case isolation was unchanged. Both incidents are preserved in `recovery_incident_initial.json` and `run_status.json`.

## Key artifacts

- `oof_predictions.csv`: all 2,784 case-OOF predictions.
- `oof_alpha_metrics.csv`, `oof_alpha_by_case.csv`, `oof_alpha_by_raw_bin.csv`: selection, stability, safety, and rescue evidence.
- `frozen_design.json`: pre-Val alpha and epoch freeze record.
- `val_predictions.csv`, `val_comparison.csv`, `val_by_case.csv`: one-shot Val results.
- `persistent_failure_summary.csv`: harmful >10px temporal-run description (analysis only; no temporal inference).
- `figures/`: alpha curves, case comparison, CDF, safety/rescue plots, and per-Series trajectories.
