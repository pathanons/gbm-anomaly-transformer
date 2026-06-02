param(
    [string]$OutputRoot = "D:\AnomalyTransformerRuns",
    [string]$DataPath = "datasets/SP500_event_taxonomy_w100",
    [string]$Device = "auto",
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [int]$WindowSize = 100,
    [int]$Step = 1,
    [int]$Seed = 42,
    [string]$Python = "python",
    [switch]$DryRun,
    [switch]$Visualize
)

$ErrorActionPreference = "Stop"
$env:AT_OUTPUT_ROOT = $OutputRoot

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

function Invoke-Experiment {
    param([string[]]$Command)
    Write-Host ($Command -join " ")
    if ($DryRun) {
        return
    }
    & $Command[0] $Command[1..($Command.Count - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed with exit code $LASTEXITCODE"
    }
}

$common = @(
    "--data-path", $DataPath,
    "--device", $Device,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--window-size", "$WindowSize",
    "--step", "$Step",
    "--seed", "$Seed",
    "--predictive-distribution", "student_t"
)
if ($Visualize) {
    $common += "--visualize"
    $common += "--show-true-labels"
}

$experiments = @(
    @{ Name = "review_none_all_studentt_rawscore"; Mode = "none"; Features = "all" },
    @{ Name = "review_temporal_all_studentt_rawscore"; Mode = "temporal"; Features = "all" },
    @{ Name = "review_logret_all_studentt_rawscore"; Mode = "gaussian_log_return"; Features = "all" },
    @{ Name = "review_canonical_all_studentt_rawscore"; Mode = "canonical_gbm"; Features = "all" },
    @{ Name = "review_none_price_studentt_rawscore"; Mode = "none"; Features = "price_only" },
    @{ Name = "review_temporal_price_studentt_rawscore"; Mode = "temporal"; Features = "price_only" },
    @{ Name = "review_logret_price_studentt_rawscore"; Mode = "gaussian_log_return"; Features = "price_only" },
    @{ Name = "review_canonical_price_studentt_rawscore"; Mode = "canonical_gbm"; Features = "price_only" }
)

foreach ($experiment in $experiments) {
    Write-Host "=== Running $($experiment.Name) ==="
    $command = @(
        $Python, "-u", "scripts/gbm/run_joint.py",
        "--exp-name", $experiment.Name,
        "--association-mode", $experiment.Mode,
        "--features", $experiment.Features
    ) + $common
    Invoke-Experiment -Command $command
}

Write-Host "=== Running predictive-distribution ablation: canonical Gaussian ==="
Invoke-Experiment -Command @(
    $Python, "-u", "scripts/gbm/run_joint.py",
    "--exp-name", "review_canonical_all_gaussian_rawscore",
    "--association-mode", "canonical_gbm",
    "--features", "all",
    "--data-path", $DataPath,
    "--device", $Device,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--window-size", "$WindowSize",
    "--step", "$Step",
    "--seed", "$Seed",
    "--predictive-distribution", "gaussian"
)

Write-Host "Review experiment suite complete. Outputs are under $OutputRoot\experiments"
