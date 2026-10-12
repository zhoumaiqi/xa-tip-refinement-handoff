$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\ProgramData\anaconda3\envs\TomoReconstruction\python.exe"
$LogPath = Join-Path $RepoRoot "outputs\phase9c_alpha_recalibration\phase9c_run.log"

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

& $PythonExe "scripts\phase9c_alpha_recalibration.py" 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) {
    throw "Phase 9C failed with exit code $LASTEXITCODE. See $LogPath"
}
