$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Source = Join-Path $Root "scripts\phase7a_causal_multiframe_crop_audit.py"
$Pyc = Join-Path $Root "scripts\__pycache__\phase7a_causal_multiframe_crop_audit.cpython-313.pyc"
$OriginalManifest = Join-Path $Root "outputs\model_clean_v1\model_clean_frames.csv"
$OutputDir = Join-Path $Root "outputs\phase7a_causal_multiframe_crop_audit"
$RuntimeManifest = Join-Path $OutputDir "phase7a_manifest_runtime.csv"

if (-not (Test-Path $Pyc)) {
    if (Test-Path $Source) {
        python -m py_compile $Source
    }
    if (-not (Test-Path $Pyc)) {
        throw "Compiled Phase 7A entry point not found: $Pyc"
    }
}

# The project-facing case identifier is case_code (for example 13424), while
# case_identifier stores a Study UID. Create a disposable audit-only manifest
# so the compiled entry point emits the intended case labels without touching
# the frozen source manifest.
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Rows = Import-Csv -LiteralPath $OriginalManifest
foreach ($Row in $Rows) { $Row.case_identifier = $Row.case_code }
$Rows | Export-Csv -LiteralPath $RuntimeManifest -NoTypeInformation -Encoding utf8

python $Pyc --manifest $RuntimeManifest --output $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Phase 7A failed with exit code $LASTEXITCODE"
}
