$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
$Entry = Join-Path $Root 'scripts\phase7d_repeat_seed_stability.src'
$Config = Join-Path $Root 'configs\phase7d_repeat_seed_stability.json'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Entry)) { throw "Phase 7D entry point missing: $Entry" }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = "$(Join-Path $Root 'work\phase5a_pydeps');$(Join-Path $Root 'work\phase3_pydeps');$(Join-Path $Root 'scripts')"
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& $Python -u $Entry --config $Config
if ($LASTEXITCODE -ne 0) { throw "Phase 7D failed with exit code $LASTEXITCODE; inspect outputs/phase7d_repeat_seed_stability/phase7d_run.log" }
