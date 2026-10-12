$ErrorActionPreference = 'Stop'
$python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = "$(Join-Path $PSScriptRoot 'work\phase5a_pydeps');$(Join-Path $PSScriptRoot 'work\phase3_pydeps');$(Join-Path $PSScriptRoot 'scripts')"
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:PYTHONUTF8 = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
Write-Host '===================================================='
Write-Host 'Phase 6G - CASE-OOF SHRINKAGE REFINEMENT'
Write-Host '===================================================='
Write-Host 'Base visual model: Phase6B heatmap'
Write-Host 'Formula: final = raw + alpha*(model-raw)'
Write-Host 'Alpha selection: 8-case OOF only'
Write-Host 'Final Val tuning: FORBIDDEN'
Write-Host 'Temporal: NONE'
Write-Host '----------------------------------------------------'
$status = Join-Path $PSScriptRoot 'outputs\phase6g_oof_shrinkage_refinement\run_status.json'
if (Test-Path -LiteralPath $status) {
    $saved = Get-Content -Raw -LiteralPath $status | ConvertFrom-Json
    if ($saved.status -eq 'complete_after_report_recovery') {
        Write-Host '[Done] Phase 6G is already complete; frozen results will not be rerun.'
        Write-Host "Alpha: $($saved.alpha_frozen_before_val)"
        Write-Host "OOF frames: $($saved.oof_frames) | one-shot Val frames: $($saved.val_frames_one_shot)"
        exit 0
    }
}
$pyc = Join-Path $PSScriptRoot 'scripts\__pycache__\phase6g_recovery_entry.cpython-313.pyc'
if (-not (Test-Path -LiteralPath $pyc)) { throw "Compiled Phase 6G driver missing: $pyc" }
$config = (Resolve-Path -LiteralPath 'configs/phase6g_oof_shrinkage_refinement.json').Path
& $python -u $pyc --config $config
if ($LASTEXITCODE -ne 0) { throw "Phase 6G failed with exit code $LASTEXITCODE; inspect outputs/phase6g_oof_shrinkage_refinement/phase6g_failure.json" }
