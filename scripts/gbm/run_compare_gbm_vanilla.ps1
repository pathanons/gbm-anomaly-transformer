<#
.SYNOPSIS
    Train canonical GBM and vanilla joint models, then compare raw test scores.

.DESCRIPTION
    Default flow:
      1) run_joint.py with --association-mode canonical_gbm
      2) run_vanilla_joint.py
      3) compare_gbm_vanilla_joint.py for one ticker or all shared tickers

    Outputs (under $OutputRoot\experiments):
      - <GbmExpName>\reports\gbm_joint_test_scores.csv
      - <VanillaExpName>\reports\vanilla_joint_test_scores.csv
      - <GbmExpName>_vs_<VanillaExpName>\ (comparison plots)

.PARAMETER CompareOnly
    Skip training; only run comparison plots (scores must already exist).

.PARAMETER RunGbmOnly
    Run only the canonical GBM pipeline.

.PARAMETER RunVanillaOnly
    Run only the vanilla pipeline.

.EXAMPLE
    .\scripts\gbm\run_compare_gbm_vanilla.ps1 -Epochs 5 -Ticker AAPL

.EXAMPLE
    .\scripts\gbm\run_compare_gbm_vanilla.ps1 -CompareOnly -GbmExpName my_gbm -VanillaExpName my_vanilla -AllTickers -Limit 5

.EXAMPLE
    .\run_compare_gbm_vanilla_full.ps1
    Train both models and plot comparison for every ticker (no limit).
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$GbmExpName = "compare_canonical_gbm",
    [string]$VanillaExpName = "compare_vanilla_joint",
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
    [string]$Ticker = "AAPL",
    [ValidateSet("anomaly", "tail", "full")]
    [string]$View = "tail",
    [int]$FocusDays = 540,
    [switch]$AllTickers,
    [int]$Limit = 0,
    [switch]$ShowTrueLabels,
    [switch]$Visualize,
    [switch]$CompareOnly,
    [switch]$RunGbmOnly,
    [switch]$RunVanillaOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:AT_OUTPUT_ROOT = $OutputRoot
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"

if ($Tickers -is [string]) {
    $Tickers = @($Tickers.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Resolve-MinicondaPython {
    param([string]$Override = "")
    if ($Override) {
        if (-not (Test-Path $Override)) { throw "Python not found: $Override" }
        return (Resolve-Path $Override).Path
    }
    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\base\python.exe"),
        (Join-Path $env:CONDA_PREFIX "python.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }
    if (-not $candidates -or @($candidates).Count -eq 0) {
        throw "Miniconda python.exe not found under $env:USERPROFILE\miniconda3"
    }
    $pythonPath = @($candidates | Select-Object -First 1)[0]
    return (Resolve-Path -LiteralPath $pythonPath).Path
}

$Python = Resolve-MinicondaPython -Override $Python
$condaBin = Split-Path $Python -Parent
$condaRoot = Split-Path $condaBin -Parent
$env:PATH = "$condaBin;$(Join-Path $condaRoot 'Scripts');$env:PATH"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host ($Command -join " ")
    if ($DryRun) {
        return
    }
    & $Command[0] $Command[1..($Command.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

$gbmCommon = @(
    "--exp-name", $GbmExpName,
    "--data-path", $DataPath,
    "--device", $Device,
    "--features", $Features,
    "--batch-size", "$BatchSize",
    "--window-size", "$WindowSize",
    "--step", "$Step",
    "--seed", "$Seed",
    "--epochs", "$Epochs",
    "--predictive-distribution", $PredictiveDistribution,
    "--association-mode", "canonical_gbm"
)

$vanillaCommon = @(
    "--exp-name", $VanillaExpName,
    "--data-path", $DataPath,
    "--device", $Device,
    "--features", $Features,
    "--batch-size", "$BatchSize",
    "--window-size", "$WindowSize",
    "--step", "$Step",
    "--seed", "$Seed",
    "--epochs", "$Epochs",
    "--prior-type", $PriorType
)

if ($Visualize) {
    $gbmCommon += "--visualize"
    $gbmCommon += "--show-true-labels"
    $vanillaCommon += "--visualize"
    $vanillaCommon += "--show-true-labels"
}

if ($Tickers -and $Tickers.Count -gt 0) {
    $gbmCommon += "--tickers"
    $vanillaCommon += "--tickers"
    foreach ($symbol in $Tickers) {
        $gbmCommon += $symbol
        $vanillaCommon += $symbol
    }
    Write-Host "Ticker subset: $($Tickers -join ', ')"
}

function Test-ScoreFile {
    param(
        [string]$ExpName,
        [string]$FileName
    )
    $path = Join-Path $OutputRoot "experiments\$ExpName\reports\$FileName"
    return (Test-Path -LiteralPath $path), $path
}

$runGbm = -not $CompareOnly -and -not $RunVanillaOnly
$runVanilla = -not $CompareOnly -and -not $RunGbmOnly
$runCompare = -not $RunGbmOnly -and -not $RunVanillaOnly

if ($runGbm) {
    Invoke-Step -Title "Canonical GBM joint pipeline" -Command @(
        $Python, "-u", "scripts/gbm/run_joint.py"
    ) + $gbmCommon
}

if ($runVanilla) {
    Invoke-Step -Title "Vanilla Anomaly Transformer pipeline" -Command @(
        $Python, "-u", "scripts/gbm/run_vanilla_joint.py"
    ) + $vanillaCommon
}

if ($runCompare) {
    $gbmOk, $gbmPath = Test-ScoreFile -ExpName $GbmExpName -FileName "gbm_joint_test_scores.csv"
    $vanillaOk, $vanillaPath = Test-ScoreFile -ExpName $VanillaExpName -FileName "vanilla_joint_test_scores.csv"
    if (-not $gbmOk -or -not $vanillaOk) {
        Write-Host ""
        Write-Host "Score files missing before compare:" -ForegroundColor Red
        if (-not $gbmOk) {
            Write-Host "  GBM:     $gbmPath" -ForegroundColor Red
        }
        if (-not $vanillaOk) {
            Write-Host "  Vanilla: $vanillaPath" -ForegroundColor Red
        }
        if ($CompareOnly) {
            throw "CompareOnly requires both score CSV files above. Re-run training first or fix -GbmExpName / -VanillaExpName."
        }
        throw "Training finished but test score CSV is missing. Check logs under $OutputRoot\experiments\<exp>\logs\"
    }

    $compareCmd = @(
        $Python, "-u", "scripts/gbm/compare_gbm_vanilla_joint.py",
        "--gbm-exp-name", $GbmExpName,
        "--vanilla-exp-name", $VanillaExpName,
        "--data-path", $DataPath,
        "--view", $View,
        "--focus-days", "$FocusDays"
    )

    if ($AllTickers) {
        $compareCmd += "--all-tickers"
        if ($Limit -gt 0) {
            $compareCmd += @("--limit", "$Limit")
        }
        $plotCount = "all shared tickers"
        if ($Limit -gt 0) {
            $plotCount = "first $Limit shared tickers"
        }
    }
    else {
        $compareCmd += @("--ticker", $Ticker)
        $plotCount = $Ticker
    }

    if ($ShowTrueLabels) {
        $compareCmd += "--show-true-labels"
    }

    Invoke-Step -Title "GBM vs vanilla score comparison plots ($plotCount)" -Command $compareCmd
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "GBM scores:     $OutputRoot\experiments\$GbmExpName\reports\gbm_joint_test_scores.csv"
Write-Host "Vanilla scores: $OutputRoot\experiments\$VanillaExpName\reports\vanilla_joint_test_scores.csv"
if ($runCompare) {
    Write-Host "Compare plots:  $OutputRoot\experiments\${GbmExpName}_vs_${VanillaExpName}\"
}
