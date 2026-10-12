$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
$Entry = Join-Path $Root 'scripts\phase9a_extended_dataset_audit.py'
$Config = Join-Path $Root 'configs\phase9a_extended_dataset_audit.json'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Entry)) { throw "Phase 9A entry point missing: $Entry" }
Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
& $Python -u $Entry --config $Config
if ($LASTEXITCODE -ne 0) { throw "Phase 9A failed with exit code $LASTEXITCODE; inspect outputs/phase9a_extended_dataset_eligibility_audit/phase9a_run.log" }
