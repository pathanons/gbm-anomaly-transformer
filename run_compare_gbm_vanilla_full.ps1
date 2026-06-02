# Full pipeline: train canonical GBM + vanilla on all tickers, plot GBM vs vanilla for every shared ticker.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Remaining
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "scripts\gbm\run_compare_gbm_vanilla_full.ps1"
if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
}

& $scriptPath @Remaining
exit $LASTEXITCODE
