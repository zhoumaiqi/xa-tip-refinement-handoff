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
