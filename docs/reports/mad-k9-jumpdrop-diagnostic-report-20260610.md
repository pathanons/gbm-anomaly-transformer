# MAD k=9 Jump/Drop Diagnostic Report

Artifact date: 2026-06-09  
Report date: 2026-06-10  
Source folder:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9
```

Primary files reviewed:

```text
final_summary.csv
107 ticker PNG files
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/reports/gbm_joint_score_metrics.json
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/reports/gbm_joint_score_summary.json
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/reports/gbm_joint_score_summary_by_event_type.csv
```

## Short Answer

The model is detecting abrupt changes in the combined robust score:

```text
robust-z(nll + association_discrepancy)
```

The k=9 MAD rule flags unusually large absolute jumps in that robust score. In this artifact, the rule works best as a conservative spike diagnostic: it reduces the number of model flags substantially compared with k=3 and makes the remaining flags much more likely to coincide with price anomaly starts.

It is not yet enough to claim final anomaly-detection superiority. The result still needs validation-selected thresholding, a unified evaluator, and fair baseline comparisons.

## What The Model Appears To Capture

The figures show that model spikes often occur near:

- abrupt price moves
- jump/drop periods
- score level shifts
- periods where the association robust-z component becomes unstable
- local transitions in the raw NLL/association score trajectory

The clearest visual pattern is not a smooth high score throughout an anomaly period. Instead, the k=9 rule catches sharp transitions in score dynamics. This makes it closer to a regime/transition detector than a dense event-window classifier.

## Aggregate Evidence From `final_summary.csv`

Across 107 ticker plots:

| Metric | Value |
| --- | ---: |
| Tickers plotted | 107 |
| Model spikes | 3,336 |
| Price anomaly starts | 2,734 |
| Model/price overlaps | 1,407 |
| Overlap per model spike | 42.18% |
| Price anomaly starts covered by overlap | 51.46% |
| Tickers with model spikes | 107 / 107 |
| Tickers with at least one overlap | 107 / 107 |

Per-ticker distribution:

| Metric | Min | Q1 | Median | Q3 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model spikes | 4 | 24 | 31 | 38 | 57 |
| Price anomaly starts | 3 | 18 | 27 | 32 | 45 |
| Overlaps | 1 | 10 | 13 | 16 | 24 |
| Overlap per model spike | 0.2500 | 0.3750 | 0.4242 | 0.4615 | 0.5789 |
| Price anomaly coverage | 0.1176 | 0.4483 | 0.5294 | 0.6250 | 0.8462 |

Interpretation:

- Median overlap-per-spike is about 42%, so the k=9 flags are not arbitrary, but they are not clean enough to be treated as final predictions.
- Median price-anomaly coverage is about 53%, so the rule catches around half of price anomaly starts per ticker.
- Every ticker has at least one overlap, which suggests the score dynamics contain broad signal across the universe.

## Comparison With k=3

The sibling k=3 artifact shows:

| MAD k | Model spikes | Price anomaly starts | Overlaps | Overlap per model spike | Coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 12,375 | 2,734 | 2,438 | 19.70% | 89.17% |
| 9 | 3,336 | 2,734 | 1,407 | 42.18% | 51.46% |

k=9 is much more conservative:

- It cuts model spikes by 73.0%.
- It raises overlap-per-spike from 19.7% to 42.2%.
- It sacrifices coverage, dropping from 89.2% to 51.5%.

This is the main reason k=9 looks better visually: it suppresses many low-confidence flags and keeps sharper score jumps. The trade-off is lower recall/coverage.

## Best-Looking Tickers

High overlap count:

| Ticker | Model spikes | Price anomaly starts | Overlaps |
| --- | ---: | ---: | ---: |
| IBM | 48 | 38 | 24 |
| FDX | 46 | 42 | 24 |
| ORCL | 47 | 45 | 22 |
| WFC | 52 | 38 | 22 |
| MRK | 57 | 33 | 21 |
| EL | 46 | 26 | 21 |
| CMCSA | 49 | 25 | 21 |
| MMM | 44 | 33 | 21 |
| UNH | 43 | 38 | 21 |
| AES | 42 | 28 | 21 |

High overlap-per-spike:

| Ticker | Model spikes | Price anomaly starts | Overlaps | Overlap per spike | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMG | 19 | 20 | 11 | 57.89% | 55.00% |
| ADBE | 34 | 32 | 19 | 55.88% | 59.38% |
| DE | 36 | 39 | 20 | 55.56% | 51.28% |
| ABBV | 13 | 10 | 7 | 53.85% | 70.00% |
| LULU | 21 | 13 | 11 | 52.38% | 84.62% |

High price-anomaly coverage:

| Ticker | Model spikes | Price anomaly starts | Overlaps | Overlap per spike | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| LULU | 21 | 13 | 11 | 52.38% | 84.62% |
| CMCSA | 49 | 25 | 21 | 42.86% | 84.00% |
| META | 10 | 6 | 5 | 50.00% | 83.33% |
| AVGO | 21 | 11 | 9 | 42.86% | 81.82% |
| EL | 46 | 26 | 21 | 45.65% | 80.77% |

## Visual Examples Reviewed

### IBM

File:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9/IBM_final_final.png
```

IBM has 48 model spikes, 38 price anomaly starts, and 24 overlaps. The figure shows many model spikes around score-level transitions and price movement changes, especially in the 2025-2026 region. This is a strong example that the delta robust-z rule can capture transition points rather than merely high score levels.

### LULU

File:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9/LULU_final_final.png
```

LULU has 21 model spikes, 13 price anomaly starts, and 11 overlaps. It is one of the best cases by coverage: 84.62% of price anomaly starts overlap with model spikes. The figure shows score regime shifts around major price movement periods.

### CMG

File:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9/CMG_final_final.png
```

CMG has 19 model spikes, 20 price anomaly starts, and 11 overlaps. This is the highest overlap-per-spike example among the reviewed summary rows: 57.89%. The rule is relatively selective while still catching more than half of price anomaly starts.

### AAPL

File:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9/AAPL_final_final.png
```

AAPL has 29 model spikes, 35 price anomaly starts, and 11 overlaps. This is a limitation case: the model score shows meaningful regime/score shifts, but the conservative k=9 threshold misses many price anomaly starts. Coverage is only 31.43%.

## Score-Level Evidence

From `gbm_joint_score_metrics.json`:

| Metric | Value |
| --- | ---: |
| Test windows | 102,855 |
| Anomaly rate | 93.36% |
| ROC-AUC from score | 0.6466 |
| PR-AUC from score | 0.9574 |
| Mean association discrepancy | 0.0987 |
| Predictive distribution | gaussian |
| Association mode | gaussian_log_return |
| Score mode | legacy |

Event-type score summaries:

| Event type | Event windows | Non-event windows | Event score mean | Non-event score mean | Event p95 | Non-event p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jump | 74,148 | 28,707 | -2.4880 | -2.6163 | -1.8229 | -1.9614 |
| drop | 79,218 | 23,637 | -2.5002 | -2.6029 | -1.8213 | -2.0112 |

The event windows have higher scores than non-event windows because the score is less negative. This supports the claim that the model score carries jump/drop signal.

Important caution: the anomaly rate is very high in this evaluation set. PR-AUC is therefore inflated by class prevalence and should not be used alone as evidence of strong detection quality.

## What Is Good About This Result

The strongest positive evidence is:

1. k=9 sharply improves flag quality compared with k=3.
2. Every plotted ticker has at least one model/price overlap.
3. Median price-anomaly coverage is above 50%.
4. Score means separate jump/drop event windows from non-event windows.
5. Visual examples show spikes around score regime shifts and price transitions.

This means the model is learning a useful financial stress signal, especially for abrupt transitions.

## What Is Not Yet Proven

This artifact does not prove:

- that k=9 is the optimal threshold
- that the model beats statistical baselines
- that the model beats forecasting/autoencoder/vanilla Anomaly Transformer baselines
- that the detector has good event-level recall under a final locked threshold
- that it provides early warning before anomaly onset
- that it generalizes across seeds or regimes

The current result is best described as:

```text
Evidence that the GBM log-return association score contains useful jump/drop transition signal, and that k=9 MAD is a cleaner conservative visualization threshold than k=3.
```

It should not yet be described as:

```text
Final proof that the proposed model is a superior anomaly detector.
```

## Recommended Next Steps

1. Recompute threshold selection on validation only, including k values such as 3, 5, 7, 9, and 11.
2. Lock the selected threshold before evaluating test performance.
3. Convert the k=9 diagnostic output into a standard `score, threshold, y_pred` evaluation file.
4. Compare against statistical baselines using the same event definitions and metrics.
5. Add missing forecasting, autoencoder/LSTM/GRU, and vanilla Anomaly Transformer baselines.
6. Report event-level recall, detection delay, false alarms per year, ROC-AUC, PR-AUC, and confidence intervals across seeds.
