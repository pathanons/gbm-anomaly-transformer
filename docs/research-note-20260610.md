# Research Note 2026-06-10

This note captures the working research discussion around the current GBM anomaly-detection pipeline, what is already implemented, what each component means, and what still needs to be validated before making publication claims.

## Current Execution Shape

The repo now uses one YAML-driven execution path:

```bash
python run.py --config <yaml>
```

Routine configs live in:

```text
configs/general/data_prepare.yaml
configs/general/train.yaml
configs/general/test.yaml
configs/general/visualize.yaml
```

Research-history configs live in `configs/phase1` through `configs/phase5`.

The active stage files are:

```text
src/gbm/datasets.py    data preparation
src/gbm/model.py       model definition
src/gbm/score.py       loss/scoring surface
src/gbm/train.py       training
src/gbm/test.py        validation/testing
src/gbm/visualize.py   k=9 MAD visualization
src/gbm/statistics.py  statistical baselines/evaluation
```

## Data

The current main dataset is:

```text
datasets/SP500_event_taxonomy_w100
```

The pipeline builds sliding windows over daily financial features, with `window_size: 100`, `step: 1`, and a chronological split. The chronological split is important because random splitting can leak future market information into training.

Training currently uses:

```yaml
include_anomalous_train: false
```

That means the model is trained primarily on normal windows, which matches the anomaly-detection setting: learn normal behavior, then flag windows that deviate.

The curated claim map indicates that the event-taxonomy labels have meaningful statistical signal. Anomaly and normal samples differ strongly in return/volatility/volume-derived features, especially volatility-related signals. This supports using the dataset as an experimental substrate, but it does not prove model superiority.

## Current Model

The active model is an Anomaly Transformer-style encoder with finance-aware association priors.

Input:

```text
[batch, window_size, features]
```

The model produces:

- reconstructed input window
- predicted return-distribution parameters such as `mu` and `sigma`
- observed return summary statistics from the input window
- attention association maps
- association discrepancy between learned attention and a selected prior

Supported association modes:

```text
none
temporal
gaussian_log_return
canonical_gbm
```

`gaussian_log_return` is the current default. `canonical_gbm` uses the canonical GBM price process and applies Ito correction in log-price drift:

```text
log drift = mu - 0.5 * sigma^2
```

The canonical GBM attention should be described as a latent source timestamp posterior over past timestamps, not merely as a standalone GBM prior.

## Loss And Scoring

The current train config uses:

```yaml
loss_mode: legacy
dist_weight: 1.0
recon_weight: 1.0
divergence_weight: 0.25
association_weight: 0.1
```

The loss combines several signals:

- reconstruction loss: abnormal windows reconstruct poorly
- predictive NLL: observed returns are unlikely under the predicted distribution
- distribution/moment discrepancy: observed and predicted return distributions disagree
- association discrepancy: learned attention differs from the finance/temporal prior

The key difference from many standard baselines is that this model does not only ask whether the input reconstructs poorly. It also asks whether the observed return dynamics are inconsistent with a learned financial return-distribution model.

## Training Protocol

Default training:

```bash
python run.py --config configs/general/train.yaml
```

Current major settings:

```yaml
batch_size: 32
epochs: 20
lr: 0.0001
device: auto
split_method: chronological
patience: 5
```

Training saves the best checkpoint according to validation loss. Validation data must be used for threshold selection, hyperparameter choice, and model selection. Test data should be used only once for final reporting.

## Current Anomaly Decision Status

The repo has score outputs and diagnostic thresholds, but it does not yet have a final unified anomaly-decision protocol for every model/baseline.

Required unified output format:

```text
ticker, start_date, end_date, split, y_true, score, threshold, y_pred
```

Required protocol:

```text
model/baseline -> score.csv -> validation threshold -> test y_pred -> final metrics
```

This is the main missing piece before making claims that one model detects anomalies better than another.

## Statistical Baselines

Statistical baselines are implemented in `src/gbm/statistics.py` and can be run with:

```bash
python run.py --config configs/phase5/statistical_baselines.yaml
```

Available statistical scores:

```text
rolling_volatility
mean_abs_return
last_return_zscore
max_return_zscore
ewma_volatility
vol_ratio_short_long
cusum_abs_return
```

These baselines do not need neural training. They compute a score from each return window. A high score means the window is suspicious. The current implementation selects a threshold from validation scores by quantile, then reports test metrics.

Interpretation:

- `rolling_volatility`: flags high realized volatility
- `mean_abs_return`: flags large average absolute return
- `last_return_zscore`: flags a final-day return jump relative to window history
- `max_return_zscore`: flags any large return jump inside the window
- `ewma_volatility`: flags recent weighted volatility
- `vol_ratio_short_long`: flags short-term volatility above long-term volatility
- `cusum_abs_return`: flags cumulative return shift

## Forecasting Error Baseline

This is needed for a fair benchmark but is not currently implemented as an active baseline after cleanup.

Training idea:

- train on normal windows
- predict next return or next OHLCV state
- select threshold using validation prediction error
- test using locked threshold

Anomaly score:

```text
score = abs(actual_return - predicted_return)
```

or, for probabilistic forecasting:

```text
score = negative log likelihood of actual return under predicted distribution
```

This baseline matters because financial anomalies should often be hard to forecast.

## Autoencoder / LSTM / GRU Baselines

These are needed for a stronger benchmark but are not currently active after repo cleanup.

Training idea:

- input the same window tensor as the proposed model
- train on normal windows
- reconstruct the input sequence
- select threshold from validation reconstruction error
- report final metrics on test

Anomaly score:

```text
score = mean squared reconstruction error
```

For localization, reconstruction error can be decomposed by timestep or feature.

These baselines matter because reconstruction-based anomaly detection is a standard comparison point.

## Vanilla Anomaly Transformer Baseline

A complete vanilla Anomaly Transformer baseline is not currently active after cleanup.

Training idea:

- use the same input windows
- use the standard Anomaly Transformer reconstruction/association objective
- do not use GBM/log-return/canonical financial priors
- select threshold using validation scores
- test under the same metrics as the proposed model

Possible anomaly score:

```text
score = reconstruction error + association discrepancy
```

or the standard vanilla formulation if reimplemented directly.

This baseline is essential because the proposed model extends the Anomaly Transformer idea with finance-aware priors. We need to show that the financial prior adds value beyond vanilla attention discrepancy.

## Proposed Model Ablations

The internal ablations are available as configs:

```bash
python run.py --config configs/phase3/association_none.yaml
python run.py --config configs/phase3/association_temporal.yaml
python run.py --config configs/phase3/association_log_return.yaml
python run.py --config configs/phase3/association_canonical.yaml
```

Interpretation:

- `none`: no useful association prior; tests whether the prior is needed at all
- `temporal`: generic time-distance prior; tests whether a simple temporal prior is enough
- `gaussian_log_return`: financial log-return prior; current default
- `canonical_gbm`: canonical GBM prior with Ito correction

Expected research question:

```text
Does gaussian_log_return or canonical_gbm outperform temporal and none?
```

If not, the claim about financial priors must be weakened.

## Robust Z-Score And k=9 MAD

The current visualization path focuses on:

```text
robust-z(nll + association_discrepancy)
abs(diff(robust-z score))
threshold = median(delta_z) + 9 * 1.4826 * MAD(delta_z)
```

Robust z-score is used because raw score components have different scales and financial data are heavy-tailed. Median and MAD are less sensitive to extreme outliers than mean and standard deviation.

k=9 MAD is currently a diagnostic threshold. It was chosen because k=3 looked too low and produced too many suspicious flags in visual inspection. k=9 is more conservative and highlights sharper score jumps.

Important limitation:

```text
k=9 is not yet a final selected threshold.
```

For a defensible experiment, k should be selected on validation data only, for example by sweeping:

```text
k in {3, 5, 7, 9, 11}
```

Then the selected k must be locked before test evaluation.

## What We Can Claim Now

Supported:

- The dataset has meaningful anomaly-label signal based on curated statistical evidence.
- The repo now has a lean YAML-driven pipeline for data preparation, training, testing, visualization, statistical baselines, and model ablations.
- Statistical baselines and proposed model ablations are partly runnable.
- The k=9 MAD visualization is useful as a conservative diagnostic display.

Not yet supported:

- The proposed model beats statistical, forecasting, autoencoder, or vanilla Anomaly Transformer baselines.
- The model provides reliable early warning.
- The financial prior is better than temporal or no prior.
- k=9 MAD is the final best threshold.
- Explanations are faithful or stable.

## Immediate Next Work

1. Build a unified evaluator that converts every score file into validation-selected thresholds, test predictions, and metrics.
2. Ensure all models/baselines export the same score schema.
3. Implement missing forecasting, autoencoder/LSTM/GRU, and vanilla Anomaly Transformer baselines.
4. Run ablations for `none`, `temporal`, `gaussian_log_return`, and `canonical_gbm`.
5. Sweep thresholds on validation only, including MAD k values.
6. Report test metrics only after threshold/model selection is locked.
7. Add multi-seed results with variance, confidence intervals, and statistical tests.

## Artifact Reading: Jump/Drop MAD k=9 Diagnostic

Reviewed artifact:

```text
D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9
```

The folder contains 107 ticker plots and `final_summary.csv`. A detailed report was added at:

```text
docs/reports/mad-k9-jumpdrop-diagnostic-report-20260610.md
```

Main finding:

```text
k=9 MAD is a conservative score-change diagnostic. It catches abrupt transitions in robust-z(nll + association_discrepancy), and many of those transitions coincide with jump/drop-style price anomaly starts.
```

Aggregate evidence from `final_summary.csv`:

| Metric | Value |
| --- | ---: |
| Tickers plotted | 107 |
| Model spikes | 3,336 |
| Price anomaly starts | 2,734 |
| Overlaps | 1,407 |
| Overlap per model spike | 42.18% |
| Price anomaly starts covered | 51.46% |
| Tickers with at least one overlap | 107 / 107 |

Compared with k=3:

| MAD k | Model spikes | Overlaps | Overlap per model spike | Price anomaly coverage |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 12,375 | 2,438 | 19.70% | 89.17% |
| 9 | 3,336 | 1,407 | 42.18% | 51.46% |

Interpretation:

- k=9 removes many low-confidence flags and makes the remaining flags more meaningful.
- The trade-off is lower coverage: the detector misses more price anomaly starts than k=3.
- Good visual examples include IBM, LULU, and CMG.
- AAPL is a useful limitation example because the score shows regime shifts but the conservative threshold covers only 31.43% of price anomaly starts.

Score-level evidence from the experiment reports:

- ROC-AUC from score: 0.6466
- PR-AUC from score: 0.9574
- anomaly rate: 93.36%
- jump event score mean is higher than non-event score mean
- drop event score mean is higher than non-event score mean

Because anomaly prevalence is very high, PR-AUC should not be over-interpreted. This artifact supports the weaker claim that the score contains useful jump/drop transition signal. It does not yet prove that the model is a final superior detector.

## Artifact Reading: Phase 2 Component Noise

Reviewed artifact:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy
```

Detailed report:

```text
docs/reports/phase2-loss-component-noise-report-20260610.md
```

Main finding:

```text
The raw legacy aggregate score is often dominated by noisy or scale-unstable components, especially reconstruction error. This supports cutting or down-weighting some terms from the final anomaly score and moving toward robustly normalized NLL + association discrepancy.
```

Key component evidence from `gbm_joint_test_scores.csv`:

| Component | Median | P99 | Max |
| --- | ---: | ---: | ---: |
| reconstruction_error | 0.122771 | 47.710999 | 7469.848145 |
| nll | -2.634786 | -1.726072 | -1.225939 |
| divergence | 0.000004 | 0.000085 | 0.000731 |
| refactored_divergence | 0.630019 | 7.529826 | 21.458315 |
| dist_qw2 | 0.000013 | 0.000312 | 0.001132 |
| dist_qw2_tail | 0.000014 | 0.000370 | 0.001324 |
| association_discrepancy | 0.411030 | 2.691291 | 3.618039 |
| total legacy score | -2.334158 | 45.238724 | 7467.566895 |

Interpretation:

- Reconstruction error has a tiny median but an extreme maximum, and the total legacy score inherits this behavior.
- NLL is much more stable and interpretable in this run.
- Association discrepancy has useful dynamic range without the same scale explosion.
- Raw divergence/QW2/QW2Tail barely move the score in this run.
- Refactored divergence moves more, but visual inspection shows it can spike noisily.

Visual examples:

- BKNG shows a total score drift into the thousands, which is scale explosion rather than clean anomaly localization.
- AAPL shows a major divergence spike and broad score hump, useful diagnostically but not a clean final decision score.
- LULU and ADBE show reconstruction/divergence spikes that drag the total score into noisy bursts.

Research implication:

```text
Cutting terms from the final score is justified when they dominate scale, create long drifts, add local peak noise, or fail to improve event localization. This evidence supports using reconstruction/divergence as ablation-controlled training or diagnostic terms, not as automatically trusted raw final-score terms.
```

Caveat:

```text
This does not prove reconstruction/divergence never help training. It only shows they are unsafe as raw final-score components without normalization, validation-selected thresholds, and ablation evidence.
```
