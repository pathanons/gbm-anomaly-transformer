<#
.SYNOPSIS
    Train canonical GBM + vanilla on all tickers, then plot raw score curves for every shared ticker.

.DESCRIPTION
    Pipeline:
      1) train + validate + test  -> canonical GBM (raw scores CSV)
      2) train + validate + test  -> vanilla joint (raw scores CSV)
      3) compare_gbm_vanilla_joint.py --all-tickers (price + GBM score + vanilla score)

    No threshold, no accuracy metrics, no per-model visualize charts unless -PerModelVisualize.

    Default Python: conda base (miniconda3\python.exe) when -Python is omitted.

.EXAMPLE
    conda activate base
    .\train_plot_all_ticker_scores.ps1 -Epochs 20

.EXAMPLE
    .\train_plot_all_ticker_scores.ps1 -CompareOnly -GbmExpName my_gbm -VanillaExpName my_vanilla
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$GbmExpName = "all_ticker_canonical_gbm",
    [string]$VanillaExpName = "all_ticker_vanilla_joint",
    [string]$Python = "",
    [string]$Device = "cuda",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [int]$Step = 1,
    [int]$Seed = 42,
    [string]$Features = "all",
    [string]$PredictiveDistribution = "student_t",
    [string]$PriorType = "gaussian",
    [ValidateSet("anomaly", "tail", "full")]
    [string]$View = "full",
    [int]$FocusDays = 540,
    [switch]$ShowTrueLabels,
    [switch]$PerModelVisualize,
    [switch]$CompareOnly,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-MinicondaPython {
    param([string]$Override = "")

    if ($Override) {
        if ($Override.StartsWith("-")) {
            throw @"
Invalid -Python value: '$Override'
Use spaces between flags, for example:
  .\train_plot_all_ticker_scores.ps1 -Python "C:\Users\Acer\miniconda3\python.exe" -Epochs 20 -Device cuda
"@
        }
        if (-not (Test-Path -LiteralPath $Override)) {
            throw "Python not found: $Override"
        }
        return (Resolve-Path -LiteralPath $Override).Path
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\base\python.exe"),
        (Join-Path $env:CONDA_PREFIX "python.exe")
    ) | Where-Object { $_ -and (Test-Path $_) }

    if (-not $candidates -or @($candidates).Count -eq 0) {
        throw @"
Miniconda python.exe not found.
Expected: $env:USERPROFILE\miniconda3\python.exe
Fix: install Miniconda, or run: conda activate base
      then pass -Python (Get-Command python).Source
"@
    }

    $pythonPath = @($candidates | Select-Object -First 1)[0]
    return (Resolve-Path -LiteralPath $pythonPath).Path
}

function Enable-MinicondaPath {
    param([string]$PythonExe)
    $condaBin = Split-Path $PythonExe -Parent
    $condaRoot = Split-Path $condaBin -Parent
    $condaScripts = Join-Path $condaRoot "Scripts"
    $pathParts = @($condaBin, $condaScripts) + ($env:PATH -split ';' | Where-Object { $_ })
    $env:PATH = ($pathParts | Select-Object -Unique) -join ';'
    $env:CONDA_PREFIX = $condaRoot
}

$Python = Resolve-MinicondaPython -Override $Python
Enable-MinicondaPath -PythonExe $Python

Write-Host "Python (miniconda): $Python" -ForegroundColor Cyan
& $Python -c "import sys, torch; print('executable', sys.executable); print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    throw "Miniconda Python cannot import torch. Run: conda activate base; pip install -r requirements.txt"
}

$mainScript = Join-Path $PSScriptRoot "run_compare_gbm_vanilla.ps1"
if (-not (Test-Path $mainScript)) {
    throw "Missing script: $mainScript"
}

$invokeArgs = @{
    OutputRoot             = $OutputRoot
    DataPath               = $DataPath
    GbmExpName             = $GbmExpName
    VanillaExpName         = $VanillaExpName
    Python                 = $Python
    Device                 = $Device
    Epochs                 = $Epochs
    BatchSize              = $BatchSize
    WindowSize             = $WindowSize
    Step                   = $Step
    Seed                   = $Seed
    Features               = $Features
    PredictiveDistribution = $PredictiveDistribution
    PriorType              = $PriorType
    View                   = $View
    FocusDays              = $FocusDays
    AllTickers             = $true
}

if ($ShowTrueLabels) {
    $invokeArgs.ShowTrueLabels = $true
}
if ($PerModelVisualize) {
    $invokeArgs.Visualize = $true
}
if ($CompareOnly) {
    $invokeArgs.CompareOnly = $true
}
if ($DryRun) {
    $invokeArgs.DryRun = $true
}

Write-Host ""
Write-Host "Train both models + plot raw scores for ALL tickers" -ForegroundColor Yellow
Write-Host "  GBM exp:     $GbmExpName"
Write-Host "  Vanilla exp: $VanillaExpName"
Write-Host "  Epochs:      $Epochs | View: $View"
Write-Host ""

& $mainScript @invokeArgs
exit $LASTEXITCODE
