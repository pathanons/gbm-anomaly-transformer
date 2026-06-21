# GBM Joint Pipeline Smoke Run

Run date: 2026-05-14  
Branch: `exp-baseline`  
Experiment: `experiment3_joint_smoke_seed42`  
Purpose: verify the active GBM joint pipeline runs end to end in the new repo.

## Command

```bat
set PYTHON=C:\Users\Acer\anaconda3\envs\anomaly-transformer-exp1\python.exe
set EXP_NAME=experiment3_joint_smoke_seed42
set EPOCHS=1
set BATCH_SIZE=64
run.bat --visualize --tickers AAPL MSFT NVDA --top-k 3
```

## Pipeline Status

The active runner completed all stages:

```text
train_joint.py -> validate_joint.py -> test_joint.py -> visualize_joint.py
```

Runtime: about 33 seconds from pipeline start to completion.  
Resolved device: `cuda`.

## Configuration

| Field | Value |
| --- | --- |
| Data path | `datasets/SP500_event_taxonomy_w100` |
| Tickers | `AAPL`, `MSFT`, `NVDA` |
| Seed | `42` |
| Window size | `100` |
| Step | `1` |
| Features | `all` |
| Epochs | `1` |
| Batch size | `64` |
| Threshold selection | validation score quantile `0.95` |

## Split Summary

| Split | Windows |
| --- | ---: |
| Train | 17,343 |
| Validation | 3,716 |
| Test | 3,718 |
| Total | 24,777 |

Per-ticker test windows:

| Ticker | Test windows |
| --- | ---: |
| AAPL | 1,355 |
| MSFT | 1,348 |
| NVDA | 1,015 |

## Validation Threshold

| Metric | Value |
| --- | ---: |
| Threshold quantile | 0.95 |
| Threshold | -1.1115566790103912 |
| Validation score mean | -2.1693398245922073 |
| Validation score std | 0.5906734023926231 |

## Test Metrics

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.5526330707721685 |
| PR-AUC | 0.9610425747816753 |
| F1 | 0.09959623149394348 |
| Precision | 0.9893048128342246 |
| Sensitivity | 0.05243764172335601 |
| Specificity | 0.9894736842105263 |
| TP | 185 |
| TN | 188 |
| FP | 2 |
| FN | 3,343 |
| Test windows | 3,718 |
| Test anomaly rate | 0.9488972565895643 |
| Mean score | -2.152403664258682 |
| Score std | 0.5909208166605731 |

## Generated Source Artifacts

The full generated outputs remain under ignored `results/` paths:

- `results/experiments/experiment3_joint_smoke_seed42/reports/gbm_joint_metrics.json`
- `results/experiments/experiment3_joint_smoke_seed42/reports/gbm_joint_threshold.json`
- `results/experiments/experiment3_joint_smoke_seed42/reports/experiment3_joint_smoke_seed42_threshold_sweep.csv`
- `results/experiments/experiment3_joint_smoke_seed42/splits/joint_split_seed42_w100_s1_all.json`
- `results/experiments/experiment3_joint_smoke_seed42/logs/joint_pipeline_20260514_222924.log`

Checkpoints, per-window score CSVs, logs, and PNG visualizations are not curated for commit.

## Interpretation Guardrail

This is a smoke run only. It verifies that the active GBM pipeline can train, validate, test, and visualize in the new repo. It does not support any claim that the GBM-aware model outperforms baselines, generalizes across seeds or regimes, or provides early warning.

The high precision and low sensitivity pattern is consistent with a conservative thresholding behavior, but the run is too small and undertrained to treat as empirical evidence.

## Threshold Sweep Follow-Up

After the initial smoke run, `scripts/gbm/sweep_validation_thresholds.py` was added to evaluate validation-selected threshold quantiles without retraining. On this same smoke run, lowering the validation quantile from `0.95` to `0.50` changed sensitivity from `0.0524` to `0.5130`, while precision remained `0.9607`. This suggests the original low sensitivity is strongly threshold-dependent, but it remains a smoke-subset observation only.
