$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = 'C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe'
$Entry = Join-Path $Root 'scripts\phase7c_temporal_domain_shift_audit.src'
$Config = Join-Path $Root 'configs\phase7c_temporal_domain_shift_audit.json'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Entry)) { throw "Phase 7C entry point missing: $Entry" }
Set-Location -LiteralPath $Root
$env:PYTHONPATH = "$(Join-Path $Root 'work\phase5a_pydeps');$(Join-Path $Root 'work\phase3_pydeps');$(Join-Path $Root 'scripts')"
$env:PYTHONUTF8 = '1'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
& $Python -u $Entry --config $Config
if ($LASTEXITCODE -ne 0) { throw "Phase 7C failed with exit code $LASTEXITCODE; inspect outputs/phase7c_temporal_domain_shift_audit/phase7c_run.log" }
