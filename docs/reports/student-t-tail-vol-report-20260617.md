# Student-t Tail/Volatility Diagnostic Report - 2026-06-17

## Setup

- Config: `configs/phase3/student_t_tail_vol_w60.yaml`
- Run: `D:/AnomalyTransformerRuns/experiments/student_t_tail_vol_w60_h4_l3_d128_lr0p0001_ep5_s42`
- Data: `datasets/SP500_event_taxonomy_w60`
- Window: 60 trading days, chronological split, purge gap 60
- Model change: `association_mode=student_t_log_return`, `predictive_distribution=student_t`
- Feature set: `log_return_tail_vol`
- Training budget: 5 epochs
- Test label mode: endpoint log-return anomaly from split/test endpoint mean +/- 3 std

## How Metrics Were Measured

- ROC-AUC: threshold-free rank metric. It checks whether anomaly windows receive higher scores than non-anomaly windows across all possible thresholds.
- PR-AUC: threshold-free precision/recall area. It is more useful than accuracy when positives are rare.
- Threshold metrics: precision, sensitivity, specificity, F1, TP/TN/FP/FN. The threshold is the validation-score p95, not selected on test.
- Component metrics: ROC/PR were also computed for raw components (`score`, `nll`, `association_discrepancy`, `tail_z_abs`) to avoid hiding weak components inside one scalar.

## Test Results

- Test windows: 111,993
- Endpoint anomaly rate: 0.013287
- Final score ROC-AUC: 0.657241
- Final score PR-AUC: 0.023271
- Validation p95 threshold: -1.77385
- Threshold F1: 0.041962
- Threshold sensitivity: 0.047715
- Threshold specificity: 0.983485
- Threshold precision: 0.037447

## Component Check

| Component | ROC-AUC | PR-AUC |
|---|---:|---:|
| score | 0.657241 | 0.023271 |
| nll | 0.655644 | 0.023106 |
| association_discrepancy | 0.750727 | 0.043521 |
| tail_z_abs | 0.670288 | 0.021094 |

## Artifacts

- `reports/test_scores.csv`
- `reports/score_metrics.json`
- `reports/evaluation_report.md`
- `reports/component_metrics.csv`
- `reports/figures/roc_curve.png`
- `reports/figures/precision_recall_curve.png`

## Interpretation

This is diagnostic evidence only. ROC-AUC is above random, but PR-AUC remains low because positives are rare and the validation p95 threshold has low sensitivity. The association-discrepancy component ranks endpoint return anomalies better than the final scalar in this run, so the next lazy ablation should test `association_discrepancy` and `tail_z_abs` scoring directly before adding heavier EVT/GARCH machinery.
