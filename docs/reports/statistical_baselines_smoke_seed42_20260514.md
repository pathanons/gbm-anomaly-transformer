# Statistical Baselines Smoke Run

Run date: 2026-05-14  
Branch: `exp-baseline`  
Experiment: `baseline_stat_smoke_seed42`  
Purpose: establish a first leakage-aware statistical baseline anchor using the same smoke-run ticker subset and split seed as the GBM joint smoke run.

## Command

```bat
C:\Users\Acer\anaconda3\envs\anomaly-transformer-exp1\python.exe -u scripts\gbm\run_statistical_baselines.py ^
  --exp-name baseline_stat_smoke_seed42 ^
  --data-path datasets/SP500_event_taxonomy_w100 ^
  --tickers AAPL MSFT NVDA ^
  --window-size 100 ^
  --step 1 ^
  --seed 42 ^
  --features all ^
  --threshold-quantile 0.95
```

## Protocol

The baseline script rebuilds the joint window manifest with the same seed and ticker subset. For each baseline score, the decision threshold is selected from validation scores only using quantile `0.95`; test windows are used only for final scoring.

Split summary:

| Split | Windows |
| --- | ---: |
| Train | 17,343 |
| Validation | 3,716 |
| Test | 3,718 |

## Test Metrics

| Baseline | ROC-AUC | PR-AUC | F1 | Precision | Sensitivity | Specificity | TP | TN | FP | FN | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rolling_volatility | 0.525119 | 0.955242 | 0.095392 | 0.967213 | 0.050170 | 0.968421 | 177 | 184 | 6 | 3351 | 0.047676 |
| mean_abs_return | 0.488285 | 0.949038 | 0.100885 | 0.944724 | 0.053288 | 0.942105 | 188 | 179 | 11 | 3340 | 0.035477 |
| last_return_zscore | 0.481115 | 0.949926 | 0.107152 | 0.975610 | 0.056689 | 0.973684 | 200 | 185 | 5 | 3328 | 2.051278 |
| max_return_zscore | 0.854076 | 0.990636 | 0.100673 | 1.000000 | 0.053005 | 1.000000 | 187 | 190 | 0 | 3341 | 6.061933 |

Shared test properties:

| Field | Value |
| --- | ---: |
| Test windows | 3,718 |
| Ticker count | 3 |
| Test anomaly rate | 0.9488972565895643 |

## Generated Source Artifacts

The full generated outputs remain under ignored `results/` paths:

- `results/experiments/baseline_stat_smoke_seed42/reports/statistical_baselines_summary.csv`
- `results/experiments/baseline_stat_smoke_seed42/reports/statistical_baselines_metrics.json`
- `results/experiments/baseline_stat_smoke_seed42/reports/max_return_zscore_smoke_threshold_sweep.csv`
- `results/experiments/baseline_stat_smoke_seed42/statistical_baselines_manifest.json`

Per-window baseline score CSVs are generated for reproducibility but are not curated for commit.

## Interpretation Guardrail

This is a smoke baseline run, not publication evidence. It uses only three tickers and one split seed. It is useful because it verifies a validation-threshold-only baseline protocol and exposes the conservative-threshold behavior against simple return/volatility scores.

No claim should be made that the GBM model beats or loses to these baselines until the comparison is repeated on the full ticker universe, multiple seeds, and a documented statistical test.

## Threshold Sweep Follow-Up

`scripts/gbm/sweep_validation_thresholds.py` can also evaluate any baseline score CSV that includes `split`, `score`, and `y_true` columns. As a smoke check, it was run for `max_return_zscore_scores.csv`; lowering the validation quantile from `0.95` to `0.50` changed sensitivity from `0.0530` to `0.5320`, with precision `0.9936`.
