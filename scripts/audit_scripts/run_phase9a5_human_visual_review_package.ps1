$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $workspace

Write-Host '===================================================='
Write-Host 'Phase 9A.5 — HUMAN VISUAL LABEL REVIEW'
Write-Host '===================================================='
Write-Host ''
Write-Host 'Purpose: Visualize EXISTING CSV labels'
Write-Host 'New annotation: NONE'
Write-Host 'Model inference: NONE'
Write-Host 'Training: NONE'
Write-Host 'Cases: 6'
Write-Host 'Series: 10'
Write-Host '----------------------------------------------------'

python scripts/phase9a5_human_visual_review_package.py --config configs/phase9a5_human_visual_review.json
if ($LASTEXITCODE -ne 0) { throw "Phase 9A.5 failed with exit code $LASTEXITCODE" }

Write-Host ''
Write-Host 'Review package ready.'
Write-Host 'Open: outputs\phase9a5_human_visual_review\review_dashboard.html'
Write-Host 'Then fill: outputs\phase9a5_human_visual_review\human_review_form.csv'
