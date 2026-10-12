$ErrorActionPreference = 'Stop'
$python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = "$(Join-Path $PSScriptRoot 'work\phase5a_pydeps');$(Join-Path $PSScriptRoot 'work\phase3_pydeps');$(Join-Path $PSScriptRoot 'scripts')"
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:PYTHONUTF8 = '1'
$phase6bArgs = @('--config', 'configs/phase6b_local_refinement_256_baseline.yaml')
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'outputs/phase6b_local_refinement_256_baseline/run_summary.json')) {
    Write-Host '[Existing completed training] Reusing best checkpoint for reporting only; no retraining.'
    $phase6bArgs += '--finalize-only'
}
& $python -u scripts/phase6b_local_refinement_256_baseline.py @phase6bArgs
if ($LASTEXITCODE -ne 0) { throw "Phase 6B failed with exit code $LASTEXITCODE; see output logs." }
