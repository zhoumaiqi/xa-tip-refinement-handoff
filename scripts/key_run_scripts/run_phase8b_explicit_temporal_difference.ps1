$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
$Entry = Join-Path $Root 'scripts\phase8b_explicit_temporal_difference.src'
$Config = Join-Path $Root 'configs\phase8b_explicit_temporal_difference.json'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Entry)) { throw "Phase 8B entry point missing: $Entry" }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = "$(Join-Path $Root 'work\phase5a_pydeps');$(Join-Path $Root 'work\phase3_pydeps');$(Join-Path $Root 'scripts')"
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& $Python -u $Entry --config $Config
if ($LASTEXITCODE -ne 0) { throw "Phase 8B failed with exit code $LASTEXITCODE; inspect outputs/phase8b_explicit_temporal_difference/phase8b_run.log" }
