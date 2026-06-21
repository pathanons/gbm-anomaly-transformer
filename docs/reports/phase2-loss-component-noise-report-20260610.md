# Phase 2 Loss/Score Component Noise Report

Report date: 2026-06-10  
Artifact folder reviewed:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy
```

Primary files reviewed:

```text
reports/gbm_joint_score_metrics.json
reports/gbm_joint_score_summary.json
reports/gbm_joint_score_metrics_by_ticker.csv
reports/gbm_joint_score_pattern_overall.json
reports/gbm_joint_score_pattern_summary_by_ticker.csv
reports/gbm_joint_test_scores.csv
visualizations_phase2_ticker_scores/*.png
```

## Short Answer

The Phase 2 score-shape artifacts support the decision to remove or heavily down-weight some loss/score components from the final anomaly score. In particular, the legacy score that includes reconstruction and moment/divergence terms is often dominated by scale explosions and long score drifts rather than clean anomaly-localized spikes.

The strongest evidence is that `reconstruction_error` has a tiny median but an extreme max, and the total legacy score inherits that same extreme max. This means a small number of reconstruction outliers can dominate the total score and behave like noise for anomaly ranking.

This supports the later cleaner diagnostic score:

```text
nll + association_discrepancy
```

with robust-z normalization and score-change thresholding.

## What Was Being Tested

This artifact corresponds to:

```text
phase2_diststudy_legacy_model_score_legacy
```

The score summary reports:

```json
{
  "dist_weight": 1.0,
  "recon_weight": 1.0,
  "divergence_weight": 0.25,
  "association_weight": 0.1,
  "score_mode": "legacy",
  "predictive_distribution": "gaussian",
  "association_mode": "gaussian_log_return"
}
```

So the legacy score mixes multiple components:

- reconstruction error
- predictive NLL
- moment/divergence discrepancy
- association discrepancy

The visualizations plot:

```text
price
total legacy score
robust-z component terms
absolute delta score
```

## Component Scale Evidence

From `gbm_joint_test_scores.csv`:

| Component | Mean | Min | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reconstruction_error | 28.895133 | 0.001733 | 0.122771 | 2.142439 | 47.710999 | 7469.848145 |
| nll | -2.615707 | -3.709988 | -2.634786 | -1.968483 | -1.726072 | -1.225939 |
| divergence | 0.000009 | 0.000000 | 0.000004 | 0.000031 | 0.000085 | 0.000731 |
| refactored_divergence | 1.176702 | 0.000003 | 0.630019 | 4.356005 | 7.529826 | 21.458315 |
| dist_qw2 | 0.000030 | 0.000000 | 0.000013 | 0.000104 | 0.000312 | 0.001132 |
| dist_qw2_tail | 0.000035 | 0.000000 | 0.000014 | 0.000121 | 0.000370 | 0.001324 |
| association_discrepancy | 0.577639 | 0.275476 | 0.411030 | 1.480571 | 2.691291 | 3.618039 |
| total legacy score | 26.337192 | -3.529510 | -2.334158 | -0.074542 | 45.238724 | 7467.566895 |

Interpretation:

- `reconstruction_error` is extremely heavy-tailed. Its median is only 0.122771, but its maximum is 7469.848145.
- The total legacy score has the same extreme behavior, with maximum 7467.566895.
- `nll` is much more stable and bounded in this run.
- `association_discrepancy` varies meaningfully but remains on a manageable scale.
- The original `divergence`, `dist_qw2`, and `dist_qw2_tail` are numerically tiny in this run and do not materially change the score.
- `refactored_divergence` has more dynamic range, but visual inspection suggests it can spike noisily and should not be trusted without validation.

This is the core reason for cutting or reducing some terms from the final anomaly score: their scale and outlier behavior can dominate the score without improving anomaly localization.

## Overall Score-Shape Evidence

From `gbm_joint_score_metrics.json`:

| Metric | Value |
| --- | ---: |
| Test windows | 102,855 |
| Anomaly rate | 98.98% |
| Score mean | 26.3372 |
| Score std | 372.2335 |
| Score median | -2.3342 |
| Score max | 7467.5669 |
| ROC-AUC from score | 0.5645 |
| PR-AUC from score | 0.9909 |

The score has a very large mean/std but a negative median. That mismatch shows the score distribution is dominated by rare extreme outliers.

The ROC-AUC is only 0.5645, so the total legacy score has weak ranking quality despite the high PR-AUC. The PR-AUC should be interpreted carefully because anomaly prevalence is 98.98%.

## Pattern Summary Evidence

From `gbm_joint_score_pattern_overall.json`:

| Metric | Value |
| --- | ---: |
| Tickers | 107 |
| Mean score std | 21.8463 |
| Mean max absolute score jump | 1.7477 |
| Mean max rolling std | 2.5472 |
| Median local peak count at q95 | 17 |

This indicates that the legacy score is often jagged or unstable enough to create many local peaks. A score with many peaks can be useful for diagnostics, but it is risky as a direct final anomaly score unless thresholded and validated carefully.

## Visual Evidence

### BKNG

File:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy/visualizations_phase2_ticker_scores/BKNG_phase2_score_shape.png
```

BKNG is the clearest failure mode. The total legacy score rises into the thousands and follows a long-scale drift rather than localized anomaly events. The by-ticker summary shows:

```text
score_mean = 3576.84
score_std = 2230.10
score_max = 7467.57
```

This is not a clean anomaly score. It is a scale-dominated reconstruction/score explosion.

### AAPL

File:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy/visualizations_phase2_ticker_scores/AAPL_phase2_score_shape.png
```

AAPL shows a major divergence robust-z spike around 2025 and a broad hump in the total score. The figure is informative diagnostically, but it does not produce a clean final decision boundary by itself. The component plot supports separating terms instead of treating the weighted sum as a final detector.

### LULU

File:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy/visualizations_phase2_ticker_scores/LULU_phase2_score_shape.png
```

LULU shows large spikes in divergence/reconstruction terms that drag the total score upward. The score then remains noisy over extended periods. This supports the decision to use score-change/robust normalization and to avoid letting reconstruction dominate the final score.

### ADBE

File:

```text
D:/AnomalyTransformerRuns/findings/phase2_diststudy_legacy_model_score_legacy/visualizations_phase2_ticker_scores/ADBE_phase2_score_shape.png
```

ADBE shows multiple component spikes and score jumps. Some align with meaningful price movement, but many create noisy score bursts. This supports using component-level diagnostics and validation-selected thresholds rather than a raw legacy aggregate score.

## Why Cutting Some Terms Makes Sense

The evidence supports cutting or reducing terms when they satisfy one or more of these conditions:

1. They dominate the scale of the total score.
2. They create long drifts rather than localized anomaly signals.
3. They produce many local peaks that do not cleanly correspond to event starts.
4. They add little numerical variation, making them irrelevant to ranking.
5. They weaken interpretability because the final score becomes a mixture of unrelated scales.

In this artifact:

- reconstruction error is the main scale-explosion risk
- raw legacy divergence is nearly zero and contributes little
- QW2 and QW2Tail are also tiny in this run and do not materially improve the score
- refactored divergence has signal but can spike noisily
- NLL is stable and interpretable
- association discrepancy is bounded enough to combine with NLL after robust normalization

Therefore, the cleaner score:

```text
nll + association_discrepancy
```

is better motivated as a diagnostic score than the raw legacy sum with reconstruction/divergence included.

## Important Caveat

This evidence supports removing or down-weighting terms from the final anomaly score. It does not automatically prove those terms should be removed from training.

Training loss and inference score should be evaluated separately:

- a term may help representation learning during training
- the same term may still be too noisy for final anomaly scoring

The next proper experiment is a loss ablation with validation-only selection:

```text
reconstruction only
nll only
association only
nll + association
nll + divergence
nll + association + divergence
full legacy
```

Then each trained model should be evaluated with the same unified validation-threshold/test-metric protocol.

## Research Position

Defensible statement:

```text
Phase 2 score-shape artifacts show that the raw legacy aggregate score is often dominated by noisy or scale-unstable components, especially reconstruction error. This motivates moving the final diagnostic score toward robustly normalized NLL plus association discrepancy and treating reconstruction/divergence as ablation-controlled terms rather than automatically trusted final-score components.
```

Not yet defensible:

```text
Reconstruction and divergence never help.
```

They may still help in some training regimes. The current evidence says they are unsafe as raw final-score components without normalization, threshold validation, and ablation support.
