param(
    [string]$Config = "configs/hrnet_phase4_5_diagnosis.yaml"
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = $PSScriptRoot
$ConfigPath = Join-Path $WorkspaceRoot $Config
$LocalDeps = Join-Path $WorkspaceRoot "work\phase3_pydeps"
$InferencePython = "C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe"
$ReportPython = "C:\ProgramData\anaconda3\python.exe"

if (-not (Test-Path -LiteralPath $InferencePython)) {
    $InferencePython = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $ReportPython)) {
    $ReportPython = $InferencePython
}
if (Test-Path -LiteralPath $LocalDeps) {
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $LocalDeps
    } else {
        $env:PYTHONPATH = "$LocalDeps;$env:PYTHONPATH"
    }
}
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=================================================="
Write-Host "Phase 4.5 Generalization Failure Diagnosis"
Write-Host "Inference/statistics only — NO TRAINING"
Write-Host "=================================================="
Write-Host "[Environment] Inference Python: $InferencePython"
Write-Host "[Environment] Config: $ConfigPath"
& $InferencePython -c "import torch; print('[Device] ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')); print('[PyTorch] ' + torch.__version__ + ' | CUDA ' + str(torch.version.cuda))"
if ($LASTEXITCODE -ne 0) {
    throw "Phase 4.5 environment check failed."
}

& $InferencePython (Join-Path $WorkspaceRoot "scripts\hrnet_phase4_5_diagnosis.py") --config $ConfigPath
if ($LASTEXITCODE -ne 0) {
    throw "Phase 4.5 inference/statistics failed with exit code $LASTEXITCODE"
}

Write-Host "[Report] Rendering figures and README in a separate no-torch process..."
& $ReportPython (Join-Path $WorkspaceRoot "scripts\hrnet_phase4_5_finalize.py") --output-dir (Join-Path $WorkspaceRoot "outputs\hrnet_phase4_5_diagnosis")
if ($LASTEXITCODE -ne 0) {
    throw "Phase 4.5 report generation failed with exit code $LASTEXITCODE"
}

Write-Host "[Done] Phase 4.5 diagnosis completed."
Write-Host "[Done] No model training or next-stage experiment was started."
Write-Host "[Done] Outputs: $(Join-Path $WorkspaceRoot 'outputs\hrnet_phase4_5_diagnosis')"
