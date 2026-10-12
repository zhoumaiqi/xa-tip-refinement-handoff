# Checkpoints

`current_baseline/phase6g_final_full_train.pth` is the engineering anchor: Phase 6G-style single-frame 256 local HRNet with historical fixed alpha 0.75.

`extended12_candidates/` contains all five prespecified Phase 9B full-train seeds. They are retained as a traceable candidate ensemble, not cherry-picked models. They were not selected using Final Test; cases 908/771 were never inferred. Phase 9C suggests alpha 0.50 as a conservative candidate for these models, but it was not promoted over the Phase 6G baseline.

OOF fold checkpoints are intentionally excluded because predictions, metrics, manifests, configs, and decisions are sufficient for audit and the fold weights are intermediate artifacts. See `CHECKPOINT_SIZE_REPORT.md`.
