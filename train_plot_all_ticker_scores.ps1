<#
.SYNOPSIS
    Train canonical GBM + vanilla (if needed), then plot raw scores for every ticker.

    By default scans D:\AnomalyTransformerRuns\experiments for existing score CSVs
    and continues from there (skips training). Override with -GbmExp / -VanillaExp.

.EXAMPLE
    .\train_plot_all_ticker_scores.ps1
    Auto-detect existing runs, plot only if both score files exist.

.EXAMPLE
    .\train_plot_all_ticker_scores.ps1 -ForceRetrain -Epochs 20

.EXAMPLE
    .\train_plot_all_ticker_scores.ps1 -GbmExp canonical_gbm_attention -VanillaExp experiment3_vanilla_joint
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$GbmExp = "",
    [string]$VanillaExp = "",
    [string]$Python = "",
    [string]$Device = "cuda",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [switch]$ForceRetrain,
    [switch]$CompareOnly
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

function Get-ExperimentInventory {
    param([string]$Root)
    $expRoot = Join-Path $Root "experiments"
    if (-not (Test-Path $expRoot)) { return @() }

    Get-ChildItem $expRoot -Directory | ForEach-Object {
        $dir = $_.FullName
        $gbmCsv = Join-Path $dir "reports\gbm_joint_test_scores.csv"
        $vanCsv = Join-Path $dir "reports\vanilla_joint_test_scores.csv"
        $gbmPt = Join-Path $dir "models\gbm_joint.pt"
        $vanPt = Join-Path $dir "models\vanilla_joint.pt"
        $gbmCsvTime = if (Test-Path $gbmCsv) { (Get-Item $gbmCsv).LastWriteTime } else { $null }
        $vanCsvTime = if (Test-Path $vanCsv) { (Get-Item $vanCsv).LastWriteTime } else { $null }

        [PSCustomObject]@{
            Name         = $_.Name
            GbmScores    = Test-Path $gbmCsv
            VanillaScores = Test-Path $vanCsv
            GbmCkpt      = Test-Path $gbmPt
            VanillaCkpt  = Test-Path $vanPt
            GbmScoresAt  = $gbmCsvTime
            VanillaScoresAt = $vanCsvTime
        }
    }
}

function Pick-GbmExperiment {
    param([array]$Inventory, [string]$Override)
    if ($Override) {
        return $Override
    }
    $candidates = $Inventory | Where-Object { $_.GbmScores }
    if (-not $candidates) { return "" }

    $preferred = @(
        "canonical_gbm_attention",
        "compare_canonical_gbm",
        "all_ticker_canonical_gbm",
        "experiment3_joint"
    )
    foreach ($name in $preferred) {
        $hit = $candidates | Where-Object { $_.Name -eq $name } | Select-Object -First 1
        if ($hit) { return $hit.Name }
    }
    $canonical = $candidates | Where-Object { $_.Name -match "canonical" } | Sort-Object GbmScoresAt -Descending | Select-Object -First 1
    if ($canonical) { return $canonical.Name }

    ($candidates | Sort-Object GbmScoresAt -Descending | Select-Object -First 1).Name
}

function Pick-VanillaExperiment {
    param([array]$Inventory, [string]$Override)
    if ($Override) {
        return $Override
    }
    $candidates = $Inventory | Where-Object { $_.VanillaScores }
    if (-not $candidates) { return "" }

    $preferred = @(
        "experiment3_vanilla_joint",
        "compare_vanilla_joint",
        "all_ticker_vanilla_joint"
    )
    foreach ($name in $preferred) {
        $hit = $candidates | Where-Object { $_.Name -eq $name } | Select-Object -First 1
        if ($hit) { return $hit.Name }
    }
    ($candidates | Sort-Object VanillaScoresAt -Descending | Select-Object -First 1).Name
}

function Invoke-PyStep {
    param([string]$Title, [string[]]$PyArgs)
    Write-Host "=== $Title ===" -ForegroundColor Yellow
    Write-Host ($PyArgs -join " ")
    & $Python @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed (exit $LASTEXITCODE)"
    }
    Write-Host ""
}

function Score-Path {
    param([string]$Exp, [string]$CsvName)
    Join-Path $OutputRoot "experiments\$Exp\reports\$CsvName"
}

# --- Discover existing runs ---
$inventory = @(Get-ExperimentInventory -Root $OutputRoot)
$GbmExp = Pick-GbmExperiment -Inventory $inventory -Override $GbmExp
$VanillaExp = Pick-VanillaExperiment -Inventory $inventory -Override $VanillaExp

Write-Host "Python:  $Python" -ForegroundColor Cyan
Write-Host "Output:  $OutputRoot" -ForegroundColor Cyan
Write-Host ""
Write-Host "Existing experiments with scores:" -ForegroundColor Cyan
if ($inventory.Count -eq 0) {
    Write-Host "  (none)"
}
else {
    $inventory | Where-Object { $_.GbmScores -or $_.VanillaScores } |
        Sort-Object Name |
        ForEach-Object {
            $parts = @()
            if ($_.GbmScores) { $parts += "GBM csv" }
            if ($_.VanillaScores) { $parts += "vanilla csv" }
            Write-Host ("  {0,-42} {1}" -f $_.Name, ($parts -join ", "))
        }
}
Write-Host ""
Write-Host "Selected GBM exp:     $GbmExp" -ForegroundColor Green
Write-Host "Selected Vanilla exp: $VanillaExp" -ForegroundColor Green
Write-Host ""

if (-not $GbmExp -or -not $VanillaExp) {
    throw "Could not resolve experiment names. Pass -GbmExp and -VanillaExp, or train first with -ForceRetrain."
}

$gbmCsv = Score-Path -Exp $GbmExp -CsvName "gbm_joint_test_scores.csv"
$vanCsv = Score-Path -Exp $VanillaExp -CsvName "vanilla_joint_test_scores.csv"
$needGbmTrain = $ForceRetrain -or (-not (Test-Path -LiteralPath $gbmCsv))
$needVanTrain = $ForceRetrain -or (-not (Test-Path -LiteralPath $vanCsv))
$skipAllTrain = $CompareOnly -or (-not $needGbmTrain -and -not $needVanTrain)

if ($skipAllTrain) {
    Write-Host "Skipping training (scores already exist). Plotting only." -ForegroundColor Magenta
    Write-Host "  GBM:     $gbmCsv"
    Write-Host "  Vanilla: $vanCsv"
    Write-Host ""
}
else {
    if ($needGbmTrain) {
        Invoke-PyStep "Train + validate + test: canonical GBM -> $GbmExp" @(
            "-u", "scripts/gbm/run_joint.py",
            "--exp-name", $GbmExp,
            "--data-path", $DataPath,
            "--device", $Device,
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--window-size", "$WindowSize",
            "--association-mode", "canonical_gbm",
            "--predictive-distribution", "student_t"
        )
    }
    if ($needVanTrain) {
        Invoke-PyStep "Train + validate + test: vanilla -> $VanillaExp" @(
            "-u", "scripts/gbm/run_vanilla_joint.py",
            "--exp-name", $VanillaExp,
            "--data-path", $DataPath,
            "--device", $Device,
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--window-size", "$WindowSize"
        )
    }
}

if (-not (Test-Path -LiteralPath $gbmCsv)) {
    throw "Missing GBM scores: $gbmCsv"
}
if (-not (Test-Path -LiteralPath $vanCsv)) {
    throw "Missing vanilla scores: $vanCsv"
}

Invoke-PyStep "Plot GBM vs vanilla (all tickers)" @(
    "-u", "scripts/gbm/compare_gbm_vanilla_joint.py",
    "--gbm-exp-name", $GbmExp,
    "--vanilla-exp-name", $VanillaExp,
    "--data-path", $DataPath,
    "--all-tickers",
    "--view", "full",
    "--show-true-labels"
)

Write-Host "Done." -ForegroundColor Green
Write-Host "Plots: $OutputRoot\experiments\${GbmExp}_vs_${VanillaExp}\"
