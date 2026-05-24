# Review Experiment Suite Summary

Audit date: 2026-05-24

External result root reviewed:

```text
D:\AnomalyTransformerRuns\experiments
```

This note summarizes the review experiment outputs created after the deep
research reports. It is a curated summary only; raw generated reports,
checkpoints, logs, per-ticker score CSVs, and figures remain outside git.

## Why These Runs Were Done

The deep research reports recommended moving away from interpreting raw
canonical GBM score as a standalone anomaly decision rule. The recommended
near-term path was:

- Keep canonical GBM as a structural or association signal rather than the
  sole anomaly criterion.
- Compare canonical GBM against weaker or simpler association alternatives.
- Use Student-t predictive distributions because financial returns are
  heavy-tailed.
- Calibrate decisions with validation-only thresholds, especially conformal or
  quantile calibration.
- Compare all-feature windows against price-only windows.
- Treat claims about baseline superiority, regime detection, and early warning
  as open until backed by dedicated evaluation.

The experiment suite addressed the first practical layer of that advice by
running controlled ablations over association mode, feature set, predictive
distribution, and calibration method.

## What Was Run

All reviewed runs used the S&P 500 event-taxonomy window benchmark with:

```text
window_size = 100
step = 1
seed = 42
device = cuda
epochs = 20
tickers = 111
train_windows = 580957
val_windows = 124431
test_windows = 102855
```

The main run families were:

| Experiment | Association mode | Features | Predictive distribution | Threshold method |
| --- | --- | --- | --- | --- |
| `review_canonical_all_studentt_conformal` | `canonical_gbm` | `all` | `student_t` | `conformal` |
| `review_canonical_all_studentt_quantile` | `canonical_gbm` | `all` | `student_t` | `quantile` |
| `review_canonical_all_gaussian_conformal` | `canonical_gbm` | `all` | `gaussian` | `conformal` |
| `review_canonical_price_studentt_conformal` | `canonical_gbm` | `price_only` | `student_t` | `conformal` |
| `review_logret_all_studentt_conformal` | `gaussian_log_return` | `all` | `student_t` | `conformal` |
| `review_logret_price_studentt_conformal` | `gaussian_log_return` | `price_only` | `student_t` | `conformal` |
| `review_temporal_all_studentt_conformal` | `temporal` | `all` | `student_t` | `conformal` |
| `review_temporal_price_studentt_conformal` | `temporal` | `price_only` | `student_t` | `conformal` |
| `review_none_all_studentt_conformal` | `none` | `all` | `student_t` | `conformal` |
| `review_none_price_studentt_conformal` | `none` | `price_only` | `student_t` | `conformal` |

Two earlier result folders were also present and inspected:

- `canonical_gbm_attention`
- `experiment3_joint`

## Headline Results

The strongest configuration by window-level F1 was:

```text
review_none_price_studentt_conformal
precision = 0.9896
sensitivity = 0.1155
specificity = 0.8832
F1 = 0.2069
ROC-AUC = 0.5242
PR-AUC = 0.9901
TP = 11759
FP = 123
TN = 930
FN = 90043
```

The best canonical GBM configuration among the reviewed runs was:

```text
review_canonical_price_studentt_conformal
precision = 0.9907
sensitivity = 0.1114
specificity = 0.8993
F1 = 0.2002
ROC-AUC = 0.5594
PR-AUC = 0.9912
TP = 11338
FP = 106
TN = 947
FN = 90464
```

The canonical all-feature Student-t conformal run was more conservative:

```text
review_canonical_all_studentt_conformal
precision = 0.9911
sensitivity = 0.0849
specificity = 0.9259
F1 = 0.1564
ROC-AUC = 0.5806
PR-AUC = 0.9915
TP = 8644
FP = 78
TN = 975
FN = 93158
```

The older `canonical_gbm_attention` run was even more conservative:

```text
precision = 0.9940
sensitivity = 0.0780
specificity = 0.9544
F1 = 0.1447
ROC-AUC = 0.5746
PR-AUC = 0.9913
```

## Interpretation

The main result is high precision but low sensitivity. These models rarely flag
relative to the number of positive windows, but when they do flag, the flags are
usually labeled anomalous. This is an operationally conservative alerting
profile, not a broad anomaly detector.

Canonical GBM did not clearly beat the simpler ablations in this single-seed
suite. The best F1 came from `none + price_only + student_t + conformal`, while
the best canonical GBM result was close but slightly lower. Therefore the
defensible statement is:

```text
Canonical GBM remains a plausible structural or association signal, but this
suite does not support claiming canonical GBM superiority over simpler
association baselines.
```

The high PR-AUC values should be interpreted carefully because the test windows
have extremely high anomaly prevalence:

```text
n_windows = 102855
anomaly_rate ~= 0.9898
```

ROC-AUC is modest across the suite, which suggests the raw ranking quality of
the current score is limited even though the calibrated decision rule can produce
very high precision.

## Event-Type Coverage

The event-type catch rates remain low. For the best F1 run,
`review_none_price_studentt_conformal`:

| Event type | Test windows with event | Catch rate |
| --- | ---: | ---: |
| `drop` | 79218 | 0.1286 |
| `jump` | 74148 | 0.1124 |
| `volatility_shock` | 68233 | 0.1267 |
| `volume_spike` | 95109 | 0.1174 |
| `regime_shift` | 0 | 0.0000 |

For `review_canonical_price_studentt_conformal`:

| Event type | Catch rate |
| --- | ---: |
| `drop` | 0.1228 |
| `jump` | 0.1156 |
| `volatility_shock` | 0.1226 |
| `volume_spike` | 0.1121 |
| `regime_shift` | 0.0000 |

This supports a cautious reading: the current decision layer catches a small,
high-confidence subset of common event windows. It does not yet establish strong
event-specific coverage, and it provides no evidence for regime-shift detection
because no regime-shift windows appear in the evaluated test set.

## What The Suite Answers

Answered or partially answered:

- Student-t and conformal/quantile calibration are wired into the active
  pipeline and produce usable outputs.
- Canonical GBM, Gaussian log-return, temporal, and no-association ablations can
  be compared under the same split.
- Price-only variants can be compared against all-feature variants.
- Validation-only thresholding is being used for the reviewed decision rules.
- Current calibrated decisions are high precision and low sensitivity.

Not answered:

- Whether canonical GBM is statistically superior to the ablations.
- Whether the model beats fair external baselines such as vanilla Anomaly
  Transformer, LSTM/GRU autoencoders, forecasting-error detectors, or rolling
  volatility/z-score rules.
- Whether results are stable across random seeds.
- Whether the score provides early warning before event onset.
- Whether the score is faithful as an explanation.
- Whether a change-point layer detects regime anomalies.
- Whether per-ticker calibration, EVT calibration, or VaR/ES/PIT/CRPS evaluation
  improves financial interpretability.

## Research Claim Status

Safe claims:

- The review suite implements the first layer of the recommended refinement
  plan: Student-t scoring, canonical/log-return/temporal/none ablations,
  price-only/all-feature variants, and validation-based calibration.
- The current decision rules produce very high precision and low sensitivity on
  the reviewed test windows.
- Canonical GBM is competitive with some ablations, especially in the
  price-only Student-t conformal setting, but is not the top F1 configuration in
  this suite.

Unsafe claims:

- Do not claim canonical GBM outperforms simpler baselines.
- Do not claim early-warning ability.
- Do not claim regime-shift detection.
- Do not claim publication-grade robustness or statistical significance.
- Do not over-interpret PR-AUC without discussing the very high anomaly
  prevalence in the test windows.

## Recommended Next Refinements

1. Add multi-seed runs for the most relevant subset:
   - `none_price_studentt_conformal`
   - `canonical_price_studentt_conformal`
   - `canonical_all_studentt_conformal`
   - `logret_all_studentt_conformal`
2. Report mean, standard deviation, confidence intervals, and paired tests.
3. Add simple external baselines under the same split:
   - rolling volatility or z-score baseline
   - forecasting-error baseline
   - recurrent autoencoder baseline
   - vanilla Anomaly Transformer baseline if feasible
4. Add per-ticker conformal calibration to test whether global calibration is
   suppressing sensitivity.
5. Add threshold sweeps or operating curves so the precision/sensitivity tradeoff
   can be selected intentionally.
6. Add event-level and lead-time metrics instead of relying only on window-level
   classification.
7. Add a change-point layer over calibrated or smoothed score if the goal is
   regime-shift detection.
8. Add financial distribution metrics if the goal shifts toward real-market risk:
   NLL, PIT, CRPS, VaR coverage, expected shortfall diagnostics, and breach
   severity.

