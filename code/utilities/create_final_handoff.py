from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAME = "XA_Tip_Refinement_Handoff_20260924"
DEST = ROOT / NAME
ZIP_PATH = ROOT / f"{NAME}.zip"


def write(rel: str, text: str) -> None:
    path = DEST / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy(src_rel: str, dst_rel: str | None = None, required: bool = True) -> bool:
    src = ROOT / src_rel
    if not src.exists():
        if required:
            raise FileNotFoundError(src)
        return False
    dst = DEST / (dst_rel or src_rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_many(src_dir: str, dst_dir: str, names: list[str]) -> None:
    for name in names:
        copy(f"{src_dir}/{name}", f"{dst_dir}/{name}", required=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


README = r"""
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
"""


STAGE_SUMMARY = r"""
# Stage summary

## Executive conclusion

The strongest supported engineering baseline is Phase 6G single-frame local refinement with fixed alpha 0.75. Later work identified temporal signal and improved safety, but neither temporal representations nor expanded training cases produced a stable primary cross-case mean-error improvement. Extended-12 with alpha 0.50 is a conservative candidate for future locked comparison, not a promoted baseline.

## Evidence chain

1. **Full-frame baseline:** fitted Train but failed held-out cases severely. Diagnostics implicated position shortcuts, visual/case domain shift, sparse-heatmap MSE weakness, and limited independent-case diversity.
2. **Local refinement:** project semantics established `raw_*` as deployable coarse localization. Raw-centered 256 crops had sufficient observed coverage and enabled a useful local HRNet.
3. **Phase 6G:** fixed alpha 0.75 gave the best reliable safety/rescue compromise available at that stage. It remains the baseline checkpoint in this handoff.
4. **Adaptive gates and residual models:** observable single-frame features did not select correction strength robustly under strict case nesting.
5. **Temporal work:** simple causal stacks repeatedly helped OOF but reversed on Val. Feature fusion failed Stage 1; explicit differences also failed independent Val stability. Temporal work remains research-only.
6. **Data expansion:** Phase 9A/9A.5 established six new technically and visually acceptable cases. Phase 9B used four for training and sealed two as Test. More cases reduced severe/persistent harm but did not reliably improve matched OOF mean.
7. **Alpha recalibration:** alpha 0.50 was preferred for Extended-12 OOF and improved tail safety, but frozen-Val mean was marginally worse; alpha is a trade-off knob, not a root-cause fix.

## Current measured anchors

- Phase 6G frozen Val: raw Mean 22.299 px; full correction about 20.008 px; fixed alpha .75 about 20.003 px, Median 13.753 px, P95 48.976 px.
- Phase 9B matched old-case OOF case-macro Mean: Baseline-8 17.044 ± 0.411 px; Extended-12 17.409 ± 0.613 px.
- Phase 9B frozen matched Val Mean: Baseline-8 21.456 ± 1.935 px; Extended-12 21.257 ± 0.590 px; descriptive only, and the OOF-C conclusion remains controlling.
- Phase 9C Extended-12: alpha .75 OOF macro 17.409 px; alpha .50 17.072 px. Frozen-Val Mean 21.257 px at .75 vs 21.315 px at .50, while P95 and severe-harm metrics improved at .50.

## What is production-relevant versus research-only

- **Engineering anchor:** Phase 6G configuration/checkpoint and raw-centered single-frame inference.
- **Traceable candidates:** five Phase 9B full-train checkpoints. They were not selected using Final Test.
- **Research-only:** temporal stacks/fusion/differences, adaptive gates, and Extended alpha .50 until an independently frozen evaluation justifies promotion.
"""


HOW_TO_RUN = r"""
# How to run

## Platform and environment

The completed experiments ran on Windows/PowerShell with Python 3.13.5 from:

`C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe`

GPU: NVIDIA RTX A2000 12GB; PyTorch 2.7.1+cu118; CUDA runtime 11.8. The default system `python` was not the training environment. See `environment/environment_info.txt` and `environment/requirements.txt`.

The copied run scripts calculate their project root from `$PSScriptRoot`; however, historical JSON configs and manifests contain absolute source paths. Update a copied config or reproduce the original directory layout before running elsewhere. Details are in `environment/path_notes.md`.

## Data root

Raw XA data were intentionally excluded. Historical source root: `C:\Users\Walter\Desktop\TIP`. Set a local data root and rebuild/verify manifests before training on a new machine. Do not point any audit/training process at reserved Test cases 908/771.

## Key entry points

From the handoff root in PowerShell:

```powershell
# Dataset eligibility audit (read-only; scans the external data root referenced by config)
.\scripts\audit_scripts\run_phase9a_extended_dataset_audit.ps1

# Original local baseline family
.\scripts\key_run_scripts\run_phase6b_local_refinement_256_baseline.ps1

# Phase 6G OOF shrinkage workflow (historical research reproduction)
.\scripts\key_run_scripts\run_phase6g_oof_shrinkage_refinement.ps1

# Extended-Train experiment; DO NOT run until external data paths and Test isolation are verified
.\scripts\key_run_scripts\run_phase9b_extended_train_singleframe.ps1

# Saved-prediction-only alpha audit; performs no model inference
.\scripts\audit_scripts\run_phase9c_alpha_recalibration.ps1
```

The historical scripts expect a repository-style layout (`scripts/`, `configs/`, `outputs/`, and external manifests). This handoff organizes code by function, so for exact reruns either restore the original layout from the folder contents or adjust the copied launcher paths. No script should be run blindly against Final Test.

## Baseline inference contract

1. Read current XA frame and current `raw_col/raw_row`.
2. Per-frame min-max normalize the full frame to [0,1].
3. Crop a 256×256 ROI centered on raw using the saved Phase 6B/6G geometry conventions.
4. Replicate grayscale into three channels.
5. Run the Phase 6G checkpoint to obtain a 64×64 heatmap.
6. Integer argmax, then map crop coordinates to original image coordinates.
7. Apply `final = raw + 0.75 × (candidate - raw)`.

Verify coordinate transforms against the saved Phase 6G predictions before using a reimplemented inference wrapper. The handoff intentionally does not invent a new unvalidated deployment wrapper.

## Evaluation

Use case-level metrics as primary evidence. Frames within a Series are correlated and must not be treated as independent patients. Report mean, median, P95/P99, easy-case harm, hard-case gain, and persistent harmful segments. Never use Final Test for threshold, alpha, seed, checkpoint, or model selection.
"""


DATA_SPLIT = r"""
# Data and split

## Coordinate and label authority

- `x = col`, `y = row`, origin at top-left, in original DICOM pixel coordinates.
- `raw_*`: product automatic coarse localization; available at deployment and permitted as crop center/input.
- `true_tip_*`: manually corrected final tip; supervision/evaluation only.
- CSV is the tip-label authority. JSON was auxiliary metadata and a JSON mismatch alone was not an exclusion reason.

## Audited inventory

The historical raw inventory contained 19 top-level cases, 57 physical XA Series, and 12,752 frames. Thirty-six Series had primary CSV labels. Raw images are external and are not duplicated here. Phase 9A inventory/mapping/eligibility tables are under `results/phase9a/`.

## Frozen partitions

- Original Train: 8 cases, 12 Series, 2,784 frames.
- Extended new Train: 4274, 14-cbct, 2768, 658; 5 Series, 1,329 frames.
- Extended total: 12 cases, 17 Series, 4,113 frames.
- Frozen Val: 13424, 2148, 6700; 4 Series, 928 frames.
- Reserved Final Test: 908, 771; 5 Series, 1,668 frames.

The six reviewed candidates were split before Phase 9B using SHA256 of `phase9b|20260924|<exact_case_id>`: first two hashes to Test and remaining four to Extended Train. The frozen split files and leakage audit are copied under `results/phase9b/` and `configs/`.

## Human review and conflict policy

All ten candidate Series across six new cases passed the user's visual review with no systematic frame-label shift, row/col swap, sustained drift, or wrong-structure labeling. Rare shadow/occlusion ambiguity remained within expected manual-label uncertainty; frames were neither deleted nor relabeled. `6700/XA-03` had conflicting annotations and remained excluded; 6700 stayed frozen Val.

## Portability

The package does not contain raw DICOM or the full 6.3 MB per-frame pixel-readability table. It includes compact inventory, mapping, eligibility, split, leakage, and candidate-manifest records. Reproduction on another machine requires reconnecting the audited external data tree and checking hashes/identities before use.
"""


FUTURE = r"""
# Future work

## Priority 1 — Failure phenotype review

Review persistent wrong-track and hard-rescue failures on Train/Val, separating wrong linear structure, diffuse/multi-peak heatmaps, border-like failures, and cases where raw was already correct. This is the most direct way to decide whether future supervision or architecture changes target the actual error.

## Priority 2 — Independent-case evidence

Continue acquiring and auditing independent cases. Phase 9B shows that four additional training cases improved safety but were not sufficient for a stable primary mean improvement. Preserve truly untouched cases rather than consuming every new case for development.

## Priority 3 — Temporal research only after evidence supports it

Temporal information showed a real OOF signal, but simple channel stacking, late fusion, and explicit difference inputs did not generalize consistently. If resumed, investigate alignment-aware or motion-aware feature fusion with more independent cases. Do not begin by simply increasing the channel stack from 3 to 5 or 7 frames.

## Before one-shot Final Test

Freeze in writing: training cases, preprocessing, model architecture, exact checkpoint/ensemble policy, seed policy, epochs, fixed alpha, handling of Series first frames, and all reported metrics. Then run exactly one prespecified evaluation on cases 908 and 771. Do not select a checkpoint or alpha from Test outcomes.
"""


TEST_WARNING = r"""
# FINAL TEST — DO NOT TOUCH

Reserved cases: **908 and 771** (5 Series, 1,668 frames).

Status through Phase 9C:

- pixel reads: 0
- GT reads: 0
- inference reads: 0
- metric reads: 0
- `test_used = false`

Identity metadata was used only for deterministic split and leakage checks. These cases must not be opened for visualization, error analysis, inference, alpha selection, checkpoint selection, thresholding, or debugging.

The Final Test may be accessed only in a separate, explicitly authorized one-shot phase after freezing training data, architecture, preprocessing, seed/ensemble policy, epoch count, alpha, checkpoint selection, and reporting metrics. A bad result must not trigger a second Test-guided model choice.
"""


EXPERIMENT_INDEX = r"""
# Experiment index

- **Phase 3/4 — Full-frame HRNet baseline.** Verified that the network could fit Train, but held-out-case error was extremely large. Established the central Train–Val generalization gap.
- **Phase 4.5 — Generalization diagnosis.** Quantified spatial/domain shift, heatmap confidence, sparse-MSE behavior, border predictions, and limited diversity. Evidence supported shortcuts and cross-case shift rather than a single code defect.
- **Phase 5A — ImageNet pretraining.** Tested pretrained initialization as a single variable; it did not eliminate cross-case failure.
- **Phase 5B — Translation augmentation.** Tested spatial augmentation in isolation; it was not a sufficient solution.
- **Phase 5C — Weighted MSE.** Increased foreground emphasis. Helpful as a local-model training choice, but not sufficient for full-frame generalization.
- **Phase 6A — Local-refinement audit.** Confirmed product semantics for raw and true tip. Showed raw-centered local crops were viable and motivated coarse-to-fine refinement.
- **Phase 6B — 256 local baseline.** Built the first practical raw-centered local HRNet and demonstrated useful refinement with important easy-case over-correction.
- **Phase 6C — Coordinate jitter.** Tested ±32 px jitter as a single variable; it was not selected as the main augmentation.
- **Phase 6D — Safe-refinement audit.** Assessed observable confidence/safety signals without training a new localizer; no simple gate was reliable enough.
- **Phase 6E — Causal temporal-consistency audit.** Characterized persistent harmful segments and showed that simple smoothing/Kalman logic would not solve wrong-track behavior.
- **Phase 6F — Case-OOF residual regression.** Improved conservatism but corrections were too small to rescue difficult cases reliably.
- **Phase 6G — Conservative heatmap shrinkage.** OOF selected fixed alpha .75, preserving visual rescue while reducing full-correction harm. This remains the engineering baseline.
- **Phase 6H — Strict nested adaptive fusion.** Used leakage-safe nested case cross-fitting. Observable gate features did not consistently beat fixed alpha .75.
- **Phase 7A — Causal multi-frame crop audit.** Confirmed legal windows, shared-current-raw crop geometry, 100% observed three-frame coverage, and no 256 padding issue.
- **Phase 7B — Causal three-frame HRNet.** Direct channel stacking improved OOF strongly but failed independent Val, revealing unstable temporal generalization.
- **Phase 7C — Temporal domain-shift audit.** Diagnosed OOF→Val reversal as multi-factor behavior including domain shift and training instability rather than a simple pipeline bug.
- **Phase 7D — Repeat-seed stability.** Across five seeds, temporal stack won OOF 5/5 by about 2.034 px, yet won Val only 1/5. The OOF temporal signal was real but not dependable across cases.
- **Phase 8A — Shared-backbone feature fusion.** Separate-frame shared HRNet plus concat/1×1 fusion failed Stage 1 (17.699 vs 17.627 px; 3/8 case wins), so Stage 2 was not run.
- **Phase 8B — Explicit temporal differences.** Signed difference channels showed some OOF value but won independent Val 0/5; this representation was not promoted.
- **Phase 9A/9A.5 — Extended-data and human review.** Rebuilt data eligibility, isolated conflicts, and generated visual review material. Six independent cases, 10 Series, and 2,997 annotated frames were accepted for controlled splitting.
- **Phase 9B — Extended-Train single-frame baseline.** Deterministic hash assigned four cases to Train and two to sealed Test. Extended-12 improved safety/persistence but won primary old-case OOF only 1/5 seeds.
- **Phase 9C — Fixed-alpha recalibration.** Saved-prediction audit selected alpha .50 for Extended-12. It improved OOF/tail safety but not frozen-Val mean, confirming a trade-off rather than a generalization fix.
"""


def build_docs() -> None:
    write("00_README_FIRST.md", README)
    write("01_STAGE_SUMMARY.md", STAGE_SUMMARY)
    write("02_HOW_TO_RUN.md", HOW_TO_RUN)
    write("03_DATA_AND_SPLIT.md", DATA_SPLIT)
    write("04_FUTURE_WORK.md", FUTURE)
    write("FINAL_TEST_DO_NOT_TOUCH.md", TEST_WARNING)
    write("experiment_history/EXPERIMENT_INDEX.md", EXPERIMENT_INDEX)


def copy_code() -> None:
    groups = {
        "code/dataset": [
            "scripts/phase2_common.py", "scripts/phase9a_extended_dataset_audit.py",
            "scripts/phase9a5_human_visual_review_package.py", "scripts/phase9b_freeze_split.py",
        ],
        "code/model": [
            "scripts/hrnet_overfit32.py", "scripts/hrnet_phase4_train.py",
            "scripts/hrnet_phase5a_train.py", "scripts/hrnet_phase5c_weighted_mse.py",
            "scripts/phase6b_local_refinement_256_baseline.py",
        ],
        "code/training": [
            "scripts/phase6g_oof_shrinkage_refinement.py", "scripts/phase7b_causal_3frame_hrnet.py",
            "scripts/phase7d_repeat_seed_stability.src", "scripts/phase8a_shared_backbone_feature_fusion.src",
            "scripts/phase8b_explicit_temporal_difference.src", "scripts/phase9b_extended_train_singleframe.py",
        ],
        "code/evaluation": [
            "scripts/phase6g_finalize_reports.py", "scripts/phase7c_temporal_domain_shift_audit.py",
            "scripts/phase8b_finalize_report.py", "scripts/phase9c_alpha_recalibration.py",
        ],
        "code/utilities": [
            "scripts/heatmap_sanity_check.py", "scripts/generate_extra_review_frames.py",
            "scripts/phase6b_render_figures.py", "scripts/phase7b_finalize_figures.py",
            "scripts/create_final_handoff.py",
        ],
    }
    for dst_dir, files in groups.items():
        for src in files:
            copy(src, f"{dst_dir}/{Path(src).name}")

    launchers = [
        "run_phase6b_local_refinement_256_baseline.ps1",
        "run_phase6g_oof_shrinkage_refinement.ps1",
        "run_phase7d_repeat_seed_stability.ps1",
        "run_phase8a_shared_backbone_feature_fusion.ps1",
        "run_phase8b_explicit_temporal_difference.ps1",
        "run_phase9b_extended_train_singleframe.ps1",
    ]
    audits = [
        "run_phase4_5_diagnosis.ps1", "run_phase6a_local_refinement_audit.ps1",
        "run_phase7a_causal_multiframe_crop_audit.ps1", "run_phase7c_temporal_domain_shift_audit.ps1",
        "run_phase9a_extended_dataset_audit.ps1", "run_phase9a5_human_visual_review_package.ps1",
        "run_phase9c_alpha_recalibration.ps1", "generate_extra_review_frames.ps1",
        "summarize_human_review.ps1",
    ]
    for src in launchers:
        copy(src, f"scripts/key_run_scripts/{src}")
    for src in audits:
        copy(src, f"scripts/audit_scripts/{src}")

    for cfg in (ROOT / "configs").glob("*"):
        if cfg.is_file():
            copy(str(cfg.relative_to(ROOT)), f"configs/{cfg.name}")


def copy_checkpoints() -> None:
    copy("outputs/phase6g_oof_shrinkage_refinement/checkpoints/final_full_train.pth",
         "checkpoints/current_baseline/phase6g_final_full_train.pth")
    copy("outputs/phase6g_oof_shrinkage_refinement/config.json",
         "checkpoints/current_baseline/phase6g_config.json")
    copy("outputs/phase6g_oof_shrinkage_refinement/frozen_design.json",
         "checkpoints/current_baseline/phase6g_frozen_design.json")
    for seed in range(20260918, 20260923):
        base = f"outputs/phase9b_extended_train_singleframe/final/seed_{seed}"
        copy(f"{base}/checkpoint.pth", f"checkpoints/extended12_candidates/seed_{seed}/checkpoint.pth")
        copy(f"{base}/complete.json", f"checkpoints/extended12_candidates/seed_{seed}/complete.json")
        copy(f"{base}/training_log.csv", f"checkpoints/extended12_candidates/seed_{seed}/training_log.csv")
    write("checkpoints/README.md", """
# Checkpoints

`current_baseline/phase6g_final_full_train.pth` is the engineering anchor: Phase 6G-style single-frame 256 local HRNet with historical fixed alpha 0.75.

`extended12_candidates/` contains all five prespecified Phase 9B full-train seeds. They are retained as a traceable candidate ensemble, not cherry-picked models. They were not selected using Final Test; cases 908/771 were never inferred. Phase 9C suggests alpha 0.50 as a conservative candidate for these models, but it was not promoted over the Phase 6G baseline.

OOF fold checkpoints are intentionally excluded because predictions, metrics, manifests, configs, and decisions are sufficient for audit and the fold weights are intermediate artifacts. See `CHECKPOINT_SIZE_REPORT.md`.
""")


def copy_results() -> None:
    selections = {
        "phase6g": (
            "outputs/phase6g_oof_shrinkage_refinement",
            ["README.md", "config.json", "frozen_design.json", "split_integrity.json", "alpha_selection.csv",
             "oof_fold_summary.csv", "oof_alpha_metrics.csv", "oof_alpha_by_case.csv", "oof_predictions.csv",
             "final_training_log.csv", "val_metrics.csv", "val_comparison.csv", "val_by_case.csv",
             "val_predictions.csv", "easy_case_safety.csv", "hard_case_rescue.csv",
             "persistent_failure_summary.csv", "run_status.json"],
            ["oof_alpha_curve.png", "val_error_cdf.png", "val_case_comparison.png"],
        ),
        "phase7d": (
            "outputs/phase7d_repeat_seed_stability",
            ["README.md", "config.json", "final_decision.json", "randomness_audit.csv", "reused_model_audit.csv",
             "oof_metrics_by_seed.csv", "oof_metrics_by_case_seed.csv", "temporal_delta_by_seed.csv",
             "temporal_case_seed_matrix.csv", "val_metrics_all_seeds.csv", "val_temporal_delta_by_seed.csv",
             "oof_vs_val_direction.csv", "persistent_harmful_by_seed.csv", "run_status.json"],
            ["oof_macro_by_seed.png", "oof_vs_val_temporal_delta.png", "temporal_case_seed_heatmap.png"],
        ),
        "phase8a": (
            "outputs/phase8a_shared_backbone_feature_fusion",
            ["README.md", "config.json", "memory_sanity.json", "parameter_count.csv", "weight_sharing_audit.txt",
             "paired_initialization_audit.csv", "artifact_integrity.csv", "run_status.json",
             "stage1/stage1_by_case.csv", "stage1/stage1_decision.json", "stage1/stage1_metrics.csv",
             "stage1/stage1_oof_D_predictions.csv", "stage1/stage1_oof_E_predictions.csv",
             "stage1/stage1_persistent_harmful.csv"],
            ["stage1_case_comparison.png"],
        ),
        "phase8b": (
            "outputs/phase8b_explicit_temporal_difference",
            ["README.md", "config.json", "final_summary.json", "terminal_summary.txt", "run_status.json",
             "stage0/difference_scale.json", "stage0/difference_statistics.csv", "stage0/difference_clipping.csv",
             "stage0/difference_tip_vs_background.csv", "stage1/metrics.csv", "stage1/by_case.csv",
             "stage1/stage1_decision.json", "stage2/metrics_by_seed.csv", "stage2/metrics_by_case_seed.csv",
             "stage2/temporal_delta_by_seed.csv", "stage2/case_seed_matrix.csv", "stage2/easy_safety_by_seed.csv",
             "stage2/hard_rescue_by_seed.csv", "stage2/persistent_by_seed.csv", "stage2/oof_decision.json",
             "stage2/val_metrics_by_seed.csv", "stage2/val_metrics_by_case_seed.csv",
             "stage2/val_delta_by_seed.csv", "stage2/oof_vs_val_direction.csv", "stage2/final_decision.json"],
            ["oof_delta_by_seed.png", "val_delta_by_seed.png", "oof_vs_val_direction.png"],
        ),
        "phase9a": (
            "outputs/phase9a_extended_dataset_eligibility_audit",
            ["README.md", "all_xa_series_inventory.csv", "dicom_duplicate_audit.csv", "csv_schema_audit.csv",
             "csv_dicom_mapping.csv", "frame_index_audit.csv", "raw_true_residual_summary.csv",
             "temporal_label_sanity.csv", "json_auxiliary_audit.csv", "annotation_conflicts.csv",
             "historical_exclusion_reconstruction.csv", "series_eligibility.csv", "case_eligibility.csv",
             "extended_dataset_candidate_manifest.csv", "manual_review_queue.csv", "expansion_scenarios.csv",
             "diversity_summary.csv", "case6700_XA03_conflict_report.md", "run_status.json"],
            ["eligibility_tier_counts.png", "eligible_frames_by_case.png", "current_vs_candidate_motion.png"],
        ),
        "phase9b": (
            "outputs/phase9b_extended_train_singleframe",
            ["README.md", "config.json", "human_review_attestation.md", "leakage_audit.csv",
             "phase9b_frozen_split.csv", "phase9b_frozen_split.json", "split_hash.txt", "test_access_audit.json",
             "train_case_frame_distribution.csv", "run_status.json", "oof/baseline_vs_extended_by_seed.csv",
             "oof/case_seed_matrix.csv", "oof/easy_safety.csv", "oof/hard_rescue.csv",
             "oof/persistent_harmful.csv", "oof/oof_decision.json", "oof/oof_metrics_by_seed.csv",
             "oof/oof_metrics_by_case_seed.csv", "final/extended_training_logs.csv",
             "oof/baseline_oof_predictions_reused.csv", "oof/extended_oof_predictions.csv",
             "final/val_baseline_predictions_reused.csv", "final/val_extended_predictions.csv",
             "final/val_delta_by_seed.csv", "final/val_metrics_by_seed.csv", "final/val_by_case_seed.csv",
             "final/val_easy_safety.csv", "final/val_hard_rescue.csv", "final/val_persistent_harmful.csv",
             "final/final_decision.json"],
            ["oof_delta_by_seed.png", "case_seed_heatmap.png", "val_delta_by_seed.png", "val_by_case.png"],
        ),
        "phase9c": (
            "outputs/phase9c_alpha_recalibration",
            ["README.md", "config.json", "candidate_availability_audit.csv", "test_access_audit.json", "run_status.json",
             "oof/alpha_grid_metrics_baseline.csv", "oof/alpha_grid_metrics_extended.csv",
             "oof/alpha_metrics_by_seed.csv", "oof/nested_alpha_choices.csv", "oof/nested_alpha_performance.csv",
             "oof/case_alpha_landscape.csv", "oof/seed_alpha_landscape.csv", "oof/easy_safety_by_alpha.csv",
             "oof/hard_rescue_by_alpha.csv", "oof/persistent_harmful_by_alpha.csv", "oof/oof_alpha_decision.json",
             "val/val_metrics_by_alpha.csv", "val/val_by_case.csv", "val/val_easy_safety.csv",
             "val/val_hard_rescue.csv", "val/val_persistent_harmful.csv", "val/final_decision.json"],
            ["oof_alpha_curve_extended.png", "alpha_choice_distribution.png", "val_alpha_comparison.png"],
        ),
    }
    for phase, (src, names, figures) in selections.items():
        copy_many(src, f"results/{phase}", names)
        for fig in figures:
            copy(f"{src}/figures/{fig}", f"results/{phase}/figures/{fig}", required=False)


def environment_docs() -> None:
    write("environment/requirements.txt", """
torch==2.7.1+cu118
torchvision==0.22.1+cu118
timm==1.0.29
pydicom==3.0.2
numpy==2.5.3
pandas==2.3.0
scipy==1.18.1
scikit-learn==1.7.2
matplotlib==3.10.3
tqdm==4.70.1
Pillow==11.0.0
""")
    write("environment/environment_info.txt", f"""
Captured: 2026-09-24 Asia/Shanghai
OS: {platform.platform()}
Training Python: 3.13.5 (conda-forge), C:\\ProgramData\\anaconda3\\envs\\TomoReconstruction\\python.exe
Default shell Python at handoff time: {sys.version.split()[0]} (not the PyTorch training environment)
GPU: NVIDIA RTX A2000 12GB
PyTorch: 2.7.1+cu118
torchvision: 0.22.1+cu118
CUDA runtime reported by PyTorch: 11.8
Primary workspace at capture: {ROOT}
Raw data root (not included): C:\\Users\\Walter\\Desktop\\TIP
""")
    write("environment/path_notes.md", r"""
# Path notes

Historical configs and CSV manifests contain absolute Windows paths to the original workspace and `C:\Users\Walter\Desktop\TIP`. The package preserves them for provenance rather than silently rewriting experimental records.

Before running elsewhere:

1. Place or mount the audited raw data separately.
2. Update copied config fields such as `manifest`, `output_dir`, and prior-phase paths.
3. Ensure the expected `scripts/`, `configs/`, and `outputs/` layout or adjust launcher paths.
4. Set `PYTHONPATH` to include the project scripts and installed packages; do not depend on the old `work/phase*_pydeps` directories.
5. Re-run split/leakage audits before training. Do not connect reserved cases 908/771 to any development inference path.

Known absolute prefixes found in the historical project include the original Codex workspace and `C:\Users\Walter\Desktop\TIP`. This handoff does not include those raw files.
""")


def checkpoint_report() -> None:
    included = list((DEST / "checkpoints").rglob("*.pth"))
    total = sum(p.stat().st_size for p in included)
    write("CHECKPOINT_SIZE_REPORT.md", "# Checkpoint size report\n\n" +
          f"Included checkpoints: {len(included)} files, {total / 1024**2:.2f} MiB total.\n\n" +
          "Included: one Phase 6G baseline and five Phase 9B Extended-12 full-train seed checkpoints. "
          "Excluded: Phase 6G fold checkpoints and Phase 9B OOF fold checkpoints, which are intermediate artifacts. "
          "No lightweight alternate ZIP was needed because the curated full package remains manageable.\n")


def make_manifest() -> None:
    rows = []
    for path in sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "HANDOFF_MANIFEST.csv"):
        rel = path.relative_to(DEST).as_posix()
        category = rel.split("/", 1)[0]
        rows.append((rel, path.stat().st_size, sha256(path), category))
    with (DEST / "HANDOFF_MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "file_size", "SHA256", "category"])
        w.writerows(rows)


def validate() -> tuple[bool, list[str]]:
    checks: list[tuple[str, bool]] = []
    required = [
        "00_README_FIRST.md", "01_STAGE_SUMMARY.md", "02_HOW_TO_RUN.md", "03_DATA_AND_SPLIT.md",
        "04_FUTURE_WORK.md", "FINAL_TEST_DO_NOT_TOUCH.md", "HANDOFF_MANIFEST.csv",
        "checkpoints/current_baseline/phase6g_final_full_train.pth",
        "results/phase9b/phase9b_frozen_split.json", "results/phase9c/oof/oof_alpha_decision.json",
    ]
    for rel in required:
        checks.append((f"required: {rel}", (DEST / rel).is_file() and (DEST / rel).stat().st_size > 0))
    checks.append(("five Extended-12 checkpoints",
                   len(list((DEST / "checkpoints/extended12_candidates").rglob("checkpoint.pth"))) == 5))
    source_pairs = [
        (ROOT / "outputs/phase6g_oof_shrinkage_refinement/checkpoints/final_full_train.pth",
         DEST / "checkpoints/current_baseline/phase6g_final_full_train.pth"),
    ]
    for seed in range(20260918, 20260923):
        source_pairs.append((ROOT / f"outputs/phase9b_extended_train_singleframe/final/seed_{seed}/checkpoint.pth",
                             DEST / f"checkpoints/extended12_candidates/seed_{seed}/checkpoint.pth"))
    checks.append(("checkpoint hashes match original artifacts",
                   all(a.exists() and b.exists() and sha256(a) == sha256(b) for a, b in source_pairs)))
    checks.append(("no raw DICOM files", not any(p.suffix.lower() in {".dcm", ".dicom"} for p in DEST.rglob("*"))))
    checks.append(("Final Test warning mentions both cases",
                   all(x in (DEST / "FINAL_TEST_DO_NOT_TOUCH.md").read_text(encoding="utf-8") for x in ("908", "771", "test_used = false"))))
    checks.append(("no zero-byte files", not any(p.is_file() and p.stat().st_size == 0 for p in DEST.rglob("*"))))
    # Validate ordinary relative Markdown links in the top-level handoff documents.
    import re
    broken_links = []
    for md in DEST.glob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            clean = target.split("#", 1)[0]
            if clean and not (md.parent / clean).exists():
                broken_links.append(f"{md.name}->{target}")
    checks.append(("top-level relative Markdown links resolve", not broken_links))
    ok = all(v for _, v in checks)
    lines = ["# Handoff validation", "", f"Overall: **{'PASS' if ok else 'FAIL'}**", ""]
    lines += [f"- [{'x' if value else ' '}] {label}" for label, value in checks]
    lines += ["", "Raw DICOM included: **NO**", "Final Test used: **NO**", "Original source/data modified: **NO**"]
    write("HANDOFF_VALIDATION.md", "\n".join(lines))
    return ok, [label for label, value in checks if not value]


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for path in sorted(p for p in DEST.rglob("*") if p.is_file()):
            zf.write(path, f"{NAME}/{path.relative_to(DEST).as_posix()}")


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    build_docs()
    copy_code()
    copy_checkpoints()
    copy_results()
    environment_docs()
    checkpoint_report()
    make_manifest()
    ok, failures = validate()
    make_manifest()  # include validation doc in the manifest
    if not ok:
        raise RuntimeError("handoff validation failed: " + "; ".join(failures))
    make_zip()
    print(json.dumps({
        "status": "PASS",
        "folder": str(DEST),
        "zip": str(ZIP_PATH),
        "folder_bytes": sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "files": sum(1 for p in DEST.rglob("*") if p.is_file()),
    }, indent=2))


if __name__ == "__main__":
    main()
