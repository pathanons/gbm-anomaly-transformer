# Launcher from repo root. Forwards all parameters to scripts/gbm/run_compare_gbm_vanilla.ps1
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$Remaining
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "scripts\gbm\run_compare_gbm_vanilla.ps1"
if (-not (Test-Path $scriptPath)) {
    throw "Missing script: $scriptPath"
}

& $scriptPath @Remaining
exit $LASTEXITCODE
