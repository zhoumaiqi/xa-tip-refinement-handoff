param(
    [string]$Config = "configs/phase6a_local_refinement_audit.yaml"
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = $PSScriptRoot
$Python = "C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Phase 6A Local Refinement Audit — statistics and visualization only, NO TRAINING"
Write-Host "Python: $Python"
Write-Host "Config: $Config"
& $Python (Join-Path $WorkspaceRoot "scripts\phase6a_local_refinement_audit.py") --config (Join-Path $WorkspaceRoot $Config)
if ($LASTEXITCODE -ne 0) { throw "Phase 6A audit failed with exit code $LASTEXITCODE" }
Write-Host "[Done] Phase 6A complete. No Phase 6B training was started."
