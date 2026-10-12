param(
    [Parameter(Mandatory=$true)][string]$Case,
    [Parameter(Mandatory=$true)][string]$Series,
    [string]$FrameRange = '',
    [int]$NumberOfFrames = 20
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $workspace
python scripts/generate_extra_review_frames.py --config configs/phase9a5_human_visual_review.json --case $Case --series $Series --frame-range $FrameRange --number-of-frames $NumberOfFrames
if ($LASTEXITCODE -ne 0) { throw "Extra review generation failed with exit code $LASTEXITCODE" }
