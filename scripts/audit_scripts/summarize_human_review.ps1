$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $workspace
$out = Join-Path $workspace 'outputs\phase9a5_human_visual_review'
$formPath = Join-Path $out 'human_review_form.csv'
$summaryPath = Join-Path $out 'human_review_summary.md'
$rows = Import-Csv -LiteralPath $formPath
$groups = $rows | Group-Object overall_decision | Sort-Object Name
$passRows = $rows | Where-Object { $_.overall_decision -eq 'PASS' }
$caseRows = $rows | Group-Object case_id | ForEach-Object {
    $statuses = @($_.Group.overall_decision | Sort-Object -Unique)
    [pscustomobject]@{
        case_id = $_.Name
        series_statuses = ($statuses -join ', ')
        case_status = if ($statuses.Count -eq 1) { $statuses[0] } else { 'MIXED' }
    }
}
$lines = @(
    '# Human visual review summary',
    '',
    'This summary reports the reviewer-entered Series decisions. It does not automatically determine final Train eligibility.',
    '',
    '## Series counts',
    ''
)
foreach ($g in $groups) { $lines += "- $($g.Name): $($g.Count) Series" }
$lines += @('', "PASS new cases represented: $(@($passRows.case_id | Sort-Object -Unique).Count)", "PASS frames: $(($passRows | Measure-Object n_frames -Sum).Sum)", '', '## Case status', '', '| Case | Series decisions | Case status |', '|---|---|---|')
foreach ($r in $caseRows) { $lines += "| $($r.case_id) | $($r.series_statuses) | $($r.case_status) |" }
$lines += @('', 'A case with mixed Series decisions is reported as MIXED and is not automatically passed.')
Set-Content -LiteralPath $summaryPath -Value $lines -Encoding UTF8
Write-Host "Saved: $summaryPath"
