# Calibration And Decision Methods

Audit date: 2026-05-24

This note records the decision-layer refinements added after reviewing the
experiment suite.

## Implemented Methods

The joint GBM pipeline now supports these threshold methods:

| Method | CLI value | Decision score | Rule |
| --- | --- | --- | --- |
| Validation quantile | `quantile` | `score` | `score > validation_normal_quantile` |
| Global conformal | `conformal` | `conformal_p_value` | `p < alpha` |
| Per-ticker conformal | `per_ticker_conformal` | `conformal_p_value` | ticker-local `p < alpha` with global fallback |
| EVT threshold | `evt` | `score` | `score > GPD tail threshold` |
| Predictive tail probability | `tail_probability` | `tail_probability` | `tail_probability < alpha` |
| VaR-style breach | `var` | `tail_probability` | `tail_probability < alpha` |

For conformal, tail-probability, and VaR-style rules:

```text
alpha = 1 - threshold_quantile
```

So the default `--threshold-quantile 0.95` corresponds to `alpha = 0.05`.

## Implementation Notes

- Shared calibration helpers live in `src/gbm/calibration.py`.
- `validate_joint.py` fits and saves threshold metadata.
- `test_joint.py` reloads threshold metadata and applies the same decision rule.
- `collect_joint_scores` now writes:
  - `tail_probability`
  - `tail_surprise = -log10(tail_probability)`
- Gaussian tail probabilities use the normal CDF.
- Student-t tail probabilities use SciPy's Student-t CDF, with a normal
  approximation fallback if SciPy is unavailable.
- EVT fits a generalized Pareto distribution to validation-normal score
  exceedances above `--evt-tail-quantile`, default `0.90`.
- If EVT fitting is not possible, the code falls back to the empirical validation
  quantile and records that fallback in the threshold JSON.

## Example Commands

Windows:

```bat
run.bat --threshold-method per_ticker_conformal --threshold-quantile 0.95
run.bat --threshold-method evt --threshold-quantile 0.95 --evt-tail-quantile 0.90
run.bat --threshold-method tail_probability --threshold-quantile 0.95
run.bat --threshold-method var --threshold-quantile 0.99
```

macOS/Linux:

```bash
DEVICE=auto bash run.sh --threshold-method per_ticker_conformal --threshold-quantile 0.95
DEVICE=auto bash run.sh --threshold-method evt --threshold-quantile 0.95 --evt-tail-quantile 0.90
DEVICE=auto bash run.sh --threshold-method tail_probability --threshold-quantile 0.95
DEVICE=auto bash run.sh --threshold-method var --threshold-quantile 0.99
```

## Recommended Next Comparison

Run the strongest existing configuration with the new decision methods:

```text
association_mode = canonical_gbm
features = price_only
predictive_distribution = student_t
```

Compare at least:

- `conformal`
- `per_ticker_conformal`
- `evt`
- `tail_probability`
- `var`

Report precision, sensitivity, F1, ROC-AUC, PR-AUC, event-type catch rates, and
per-ticker metric spread. Do not select the final method from test results
alone; use validation behavior or a pre-registered operating point first.

## Decision Sweep Runner

Use `scripts/gbm/run_decision_sweep.py` to vary decision-layer values from an
existing trained checkpoint without retraining the model.

Windows:

```bat
python -u scripts\gbm\run_decision_sweep.py --config configs\train1\review_decision_sweep_canonical_price.yaml
```

The example config pins `python: C:\Users\Acer\anaconda3\python.exe` because the
system Python on this machine does not have torch installed. Override with
`--python <path>` if using another environment.

Equivalent explicit Windows command:

```bat
python -u scripts\gbm\run_decision_sweep.py ^
  --python C:\Users\Acer\anaconda3\python.exe ^
  --checkpoint-exp-name review_canonical_price_studentt_conformal ^
  --sweep-name review_canonical_price_studentt_decision_sweep ^
  --features price_only ^
  --association-mode canonical_gbm ^
  --predictive-distribution student_t ^
  --threshold-methods conformal,per_ticker_conformal,evt,tail_probability,var ^
  --threshold-quantiles 0.90,0.95,0.975,0.99 ^
  --evt-tail-quantiles 0.85,0.90,0.95
```

macOS/Linux:

```bash
python -u scripts/gbm/run_decision_sweep.py --config configs/train1/review_decision_sweep_canonical_price.yaml
```

Equivalent explicit macOS/Linux command:

```bash
python -u scripts/gbm/run_decision_sweep.py \
  --checkpoint-exp-name review_canonical_price_studentt_conformal \
  --sweep-name review_canonical_price_studentt_decision_sweep \
  --features price_only \
  --association-mode canonical_gbm \
  --predictive-distribution student_t \
  --threshold-methods conformal,per_ticker_conformal,evt,tail_probability,var \
  --threshold-quantiles 0.90,0.95,0.975,0.99 \
  --evt-tail-quantiles 0.85,0.90,0.95
```

This creates separate experiment folders for each decision configuration and
writes an aggregate table to:

```text
D:\AnomalyTransformerRuns\experiments\<sweep_name>\reports\decision_sweep_summary.csv
```
