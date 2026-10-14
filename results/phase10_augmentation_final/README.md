# Phase 10 Data Augmentation

This is the cleaned final package for the augmentation study.

## Keep

- `checkpoints/`: saved all-development-train `A2_jitter8` candidate model
- `final_candidate_status.json`: model path and data-isolation record
- `phase10_augmentation_report.md`: OOF screening and multi-seed conclusion
- `oof_condition_summary.csv`, `oof_metrics_by_condition.csv`: OOF summaries
- `frozen_val_confirmation/`: one-time Frozen Val metrics, status, and report
- `development_manifest.csv`, `manifest_summary.csv`, `manifest_hash.txt`:
  reproducible data index and split summary
- `visualizations/`: one representative epoch-1 image per sampled frame for
  `A0_none`, `A1_intensity`, `A2_jitter8`, `A2_jitter16`, and `A2_jitter32`

## Important Conclusion

`A2_jitter8` improved average OOF and Frozen Val localization error, but on
Frozen Val it increased severe harmful corrections. The checkpoint is retained
as a research candidate, not as the default safe model.

Frozen Val has been used once for this decision. Final Test cases `908` and
`771` remain unused.
