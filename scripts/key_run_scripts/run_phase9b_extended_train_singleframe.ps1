$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $workspace
$python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
$env:PYTHONPATH = "$(Join-Path $workspace 'work\phase5a_pydeps');$(Join-Path $workspace 'work\phase3_pydeps');$(Join-Path $workspace 'scripts')"
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

Write-Host '===================================================='
Write-Host 'Phase 9B — EXTENDED-TRAIN SINGLE-FRAME BASELINE'
Write-Host '===================================================='
Write-Host ''
Write-Host 'Scientific variable: TRAINING DATA EXPANSION'
Write-Host 'Model: UNCHANGED'
Write-Host 'Temporal: NONE'
Write-Host 'Current Train: 8 cases'
Write-Host 'New reviewed cases: 6'
Write-Host 'Hash split: 4 → Extended Train / 2 → Reserved Final Test'
Write-Host 'Frozen Val: 13424 / 2148 / 6700'
Write-Host 'Final Test: RESERVED / NO INFERENCE'
Write-Host 'Seeds: 5'
Write-Host 'Alpha: 0.75 fixed'
Write-Host '----------------------------------------------------'

& $python scripts/phase9b_freeze_split.py --config configs/phase9b_extended_train_singleframe.json
if ($LASTEXITCODE -ne 0) { throw "Split verification failed with exit code $LASTEXITCODE" }
& $python -u scripts/phase9b_extended_train_singleframe.py --config configs/phase9b_extended_train_singleframe.json
if ($LASTEXITCODE -ne 0) { throw "Phase 9B failed with exit code $LASTEXITCODE" }
