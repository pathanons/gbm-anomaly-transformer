<#
.SYNOPSIS
    Experiment 2: GBM score component ablation (reconstruction / KL / mixed).
    Recomputes raw scores from stored test components (no retraining, no thresholds),
    saves to a separate experiment folder, then plots every variant for every ticker.

.EXAMPLE
    .\run_loss_ablation_experiment.ps1

.EXAMPLE
    .\run_loss_ablation_experiment.ps1 -SourceExp canonical_gbm_attention -OutputExp score_component_ablation

.EXAMPLE
    .\run_loss_ablation_experiment.ps1 -PlotOnly -Variants "recon_only,association_kl_only,full_default"
#>
param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$SourceExp = "canonical_gbm_attention",
    [string]$OutputExp = "score_component_ablation",
    [string]$Variants = "recon_only,association_kl_only,recon_plus_kl,nll_only,full_default",
    [string]$Python = "",
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

if (-not $PlotOnly) {
    $sourceScores = Join-Path $OutputRoot "experiments\$SourceExp\reports\gbm_joint_test_scores.csv"
    if (-not (Test-Path $sourceScores)) {
        throw "Missing source GBM scores: $sourceScores`nRun canonical GBM test first."
    }

    Write-Host "[loss_ablation] source=$SourceExp -> output=$OutputExp" -ForegroundColor Cyan
    & $Python scripts/gbm/run_score_component_ablation.py `
        --source-exp-name $SourceExp `
        --output-exp-name $OutputExp `
        --variants $Variants
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$plotArgs = @(
    "scripts/gbm/plot_score_component_ablation.py",
    "--output-exp-name", $OutputExp,
    "--variants", $Variants,
    "--data-path", $DataPath,
    "--all-tickers"
)
if ($ShowTrueLabels) { $plotArgs += "--show-true-labels" }

Write-Host "[loss_ablation] plot score variants (all tickers)" -ForegroundColor Cyan
& $Python @plotArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done." -ForegroundColor Green
Write-Host "  Scores : $OutputRoot\experiments\$OutputExp\reports"
Write-Host "  Figures: $OutputRoot\experiments\$OutputExp\figures\score_variants"
