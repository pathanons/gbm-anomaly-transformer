<#
.SYNOPSIS
    Run statistical, neural (LSTM/MLP AE), and sklearn (RF/IsolationForest) baselines
    on the same GBM window protocol, then optionally compare plots vs canonical GBM.

.EXAMPLE
    .\run_baselines.ps1

.EXAMPLE
    .\run_baselines.ps1 -Baselines "rolling_volatility,lstm_autoencoder,random_forest" -Epochs 10 -Device cuda

.EXAMPLE
    .\run_baselines.ps1 -CompareGbm -GbmExp canonical_gbm_attention -Baseline lstm_autoencoder -AllTickers
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$ExpName = "baseline_comparison",
    [string]$GbmExp = "canonical_gbm_attention",
    [string]$Baselines = "all",
    [string]$Baseline = "lstm_autoencoder",
    [string]$Python = "",
    [string]$Device = "cuda",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [switch]$SkipExisting,
    [switch]$CompareGbm,
    [switch]$AllTickers,
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

$runArgs = @{
    "exp-name"   = $ExpName
    "data-path"  = $DataPath
    "window-size" = $WindowSize
    "batch-size" = $BatchSize
    "epochs"     = $Epochs
    "device"     = $Device
    "baselines"  = $Baselines
}
if ($SkipExisting) {
    $runArgs["skip-existing"] = $true
}

$argList = @()
foreach ($key in $runArgs.Keys) {
    $argList += "--$key"
    $argList += [string]$runArgs[$key]
}

Write-Host "[run_baselines] python=$Python exp=$ExpName baselines=$Baselines" -ForegroundColor Cyan
& $Python "scripts/gbm/run_baselines.py" @argList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($CompareGbm) {
    $compareArgs = @(
        "scripts/gbm/compare_gbm_baselines.py",
        "--gbm-exp-name", $GbmExp,
        "--baseline-exp-name", $ExpName,
        "--baseline", $Baseline
    )
    if ($AllTickers) { $compareArgs += "--all-tickers" }
    if ($ShowTrueLabels) { $compareArgs += "--show-true-labels" }
    Write-Host "[run_baselines] compare GBM vs $Baseline" -ForegroundColor Cyan
    & $Python @compareArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Done. Reports: $OutputRoot\experiments\$ExpName\reports" -ForegroundColor Green
