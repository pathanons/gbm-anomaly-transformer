<#
.SYNOPSIS
    Experiment 1: Run all baselines into a dedicated experiment folder, then plot
    raw GBM scores vs every baseline for every ticker (no thresholds).

.EXAMPLE
    .\run_baseline_experiment.ps1

.EXAMPLE
    .\run_baseline_experiment.ps1 -ExpName baseline_comparison -Device cuda -Epochs 20 -SkipExisting

.EXAMPLE
    .\run_baseline_experiment.ps1 -PlotOnly -GbmExp canonical_gbm_attention
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$ExpName = "baseline_comparison",
    [string]$GbmExp = "canonical_gbm_attention",
    [string]$Baselines = "all",
    [string]$Python = "",
    [string]$Device = "cuda",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [switch]$SkipExisting,
    [switch]$PlotOnly,
    [switch]$ShowTrueLabels
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Python) {
    $Python = "$env:USERPROFILE\miniconda3\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$condaBin = Split-Path $Python -Parent
$condaRoot = Split-Path $condaBin -Parent
$env:PATH = "$condaBin;$(Join-Path $condaRoot 'Scripts');$env:PATH"
$env:AT_OUTPUT_ROOT = $OutputRoot
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"

$gbmScores = Join-Path $OutputRoot "experiments\$GbmExp\reports\gbm_joint_test_scores.csv"
if (-not (Test-Path $gbmScores)) {
    throw "Missing GBM scores for compare plots: $gbmScores`nTrain/test canonical GBM first."
}

if (-not $PlotOnly) {
    $runArgs = @(
        "scripts/gbm/run_baselines.py",
        "--exp-name", $ExpName,
        "--data-path", $DataPath,
        "--window-size", "$WindowSize",
        "--batch-size", "$BatchSize",
        "--epochs", "$Epochs",
        "--device", $Device,
        "--baselines", $Baselines
    )
    if ($SkipExisting) { $runArgs += "--skip-existing" }

    Write-Host "[baseline_experiment] run baselines -> $ExpName" -ForegroundColor Cyan
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$compareArgs = @(
    "scripts/gbm/compare_all_baselines.py",
    "--gbm-exp-name", $GbmExp,
    "--baseline-exp-name", $ExpName,
    "--data-path", $DataPath,
    "--all-tickers"
)
if ($ShowTrueLabels) { $compareArgs += "--show-true-labels" }

Write-Host "[baseline_experiment] plot GBM vs baselines (all tickers)" -ForegroundColor Cyan
& $Python @compareArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done." -ForegroundColor Green
Write-Host "  Scores : $OutputRoot\experiments\$ExpName\reports"
Write-Host "  Figures: $OutputRoot\experiments\$ExpName\figures\gbm_vs_baselines"
