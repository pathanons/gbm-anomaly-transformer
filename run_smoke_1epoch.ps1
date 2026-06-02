# Quick smoke test: 1 epoch, 2 tickers, train both models, compare plots.
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$Python = "",  # default: %USERPROFILE%\miniconda3\python.exe via train_plot script
    [string[]]$Tickers = @("AAPL", "MSFT"),
    [int]$Epochs = 1,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $condaBase = Join-Path $env:USERPROFILE "miniconda3\python.exe"
    if (Test-Path $condaBase) {
        $Python = $condaBase
    }
    else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}
Write-Host "Using Python: $Python"

$repoRoot = $PSScriptRoot
$fullScript = Join-Path $repoRoot "scripts\gbm\run_compare_gbm_vanilla_full.ps1"

& $fullScript `
    -OutputRoot $OutputRoot `
    -Python $Python `
    -Epochs $Epochs `
    -Device $Device `
    -GbmExpName "smoke_1ep_canonical_gbm" `
    -VanillaExpName "smoke_1ep_vanilla_joint" `
    -Tickers $Tickers `
    -BatchSize 32

exit $LASTEXITCODE
