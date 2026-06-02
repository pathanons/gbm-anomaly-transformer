<#
.SYNOPSIS
    Full run: train canonical GBM + vanilla on all dataset tickers, then plot every shared ticker.

.DESCRIPTION
    Same as run_compare_gbm_vanilla.ps1 but always:
      - trains both models on all tickers found under DataPath
      - runs compare_gbm_vanilla_joint.py with --all-tickers (no limit)
      - shows true label spans on plots

    Per-ticker joint/vanilla visualize.py charts are OFF by default (very slow for ~100 tickers).
    Use -PerModelVisualize to enable them.

.EXAMPLE
    .\run_compare_gbm_vanilla_full.ps1

.EXAMPLE
    .\run_compare_gbm_vanilla_full.ps1 -Epochs 20 -OutputRoot D:\AnomalyTransformerRuns
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$GbmExpName = "full_canonical_gbm",
    [string]$VanillaExpName = "full_vanilla_joint",
    [string]$Device = "auto",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [int]$Step = 1,
    [int]$Seed = 42,
    [string]$Features = "all",
    [string]$PredictiveDistribution = "student_t",
    [string]$PriorType = "gaussian",
    [string]$Python = "",
    [string[]]$Tickers = @(),
    [ValidateSet("anomaly", "tail", "full")]
    [string]$View = "tail",
    [int]$FocusDays = 540,
    [switch]$PerModelVisualize,
    [switch]$CompareOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:AT_OUTPUT_ROOT = $OutputRoot
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

function Get-TickerCount {
    param([string]$Path)
    $resolved = Join-Path $root $Path
    if (-not (Test-Path $resolved)) {
        throw "Data path not found: $resolved"
    }
    return @(Get-ChildItem -Path $resolved -Filter "*_ohlcv.csv" -File).Count
}

$tickerCount = Get-TickerCount -Path $DataPath
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host " GBM vs Vanilla - FULL ALL-TICKER RUN" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Data path:        $DataPath"
Write-Host "OHLCV tickers:    $tickerCount"
Write-Host "Output root:      $OutputRoot"
Write-Host "GBM experiment:   $GbmExpName"
Write-Host "Vanilla experiment: $VanillaExpName"
Write-Host "Epochs:           $Epochs | Window: $WindowSize | Batch: $BatchSize"
Write-Host "Compare view:     $View | Focus days: $FocusDays"
Write-Host "Comparison plots: ALL shared tickers (no limit)"
if ($CompareOnly) {
    Write-Host "Mode:             CompareOnly (skip training)" -ForegroundColor Magenta
}
if ($DryRun) {
    Write-Host "Mode:             DryRun" -ForegroundColor Magenta
}
Write-Host ""

if ($Tickers -is [string]) {
    $Tickers = @($Tickers.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
if ($Tickers -and $Tickers.Count -gt 0) {
    Write-Host "Training/plot subset: $($Tickers -join ', ')"
}

$mainScript = Join-Path $PSScriptRoot "run_compare_gbm_vanilla.ps1"
$invokeArgs = @{
    OutputRoot             = $OutputRoot
    DataPath               = $DataPath
    GbmExpName             = $GbmExpName
    VanillaExpName         = $VanillaExpName
    Device                 = $Device
    Epochs                 = $Epochs
    BatchSize              = $BatchSize
    WindowSize             = $WindowSize
    Step                   = $Step
    Seed                   = $Seed
    Features               = $Features
    PredictiveDistribution = $PredictiveDistribution
    PriorType              = $PriorType
    Python                 = $Python
    View                   = $View
    FocusDays              = $FocusDays
    AllTickers             = $true
    ShowTrueLabels         = $true
}
if ($CompareOnly) {
    $invokeArgs.CompareOnly = $true
}
if ($PerModelVisualize) {
    $invokeArgs.Visualize = $true
}
if ($DryRun) {
    $invokeArgs.DryRun = $true
}
if ($Tickers -and $Tickers.Count -gt 0) {
    $invokeArgs.Tickers = $Tickers
}

& $mainScript @invokeArgs
exit $LASTEXITCODE
