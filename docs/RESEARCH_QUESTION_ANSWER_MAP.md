# Research Question And Answer Map

Audit date: 2026-05-14

This file converts the current project into a research-question map. It distinguishes answered questions, partially answered questions, and questions that remain open. Evidence is limited to curated artifacts currently present in this repository.

Important scope note: there are no fresh `results/experiments` outputs in this repo at the time of this audit. The evidence below comes from curated legacy imports under `docs/legacy/`.

## Evidence Index

| ID | Evidence | Path | Artifact date |
| --- | --- | --- | --- |
| E1 | Dataset validation and leakage/signal insight | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP1__insight.md` | 2026-04-13 23:00:09 +07:00 |
| E2 | Power-law prior diagnostic result | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md` | 2026-04-15 14:22:12 +07:00 |
| E3 | GBM-implied distribution consistency experiment plan | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP3__experiment_plan.md` | 2026-04-16 11:46:52 +07:00 |
| E4 | Score-change dynamic threshold metrics, rolling window 50, MAD k=3.5 | `docs/legacy/gbm-report-summaries/experiment4_score_change_dynamic_threshold/reports/score_change_dynamic_threshold_metrics.json` | 2026-05-04 20:51:06 +07:00 |
| E5 | Score-change dynamic threshold metrics, rolling window 20, MAD k=1.5 | `docs/legacy/gbm-report-summaries/experiment4_score_change_dynamic_threshold_w20_k1p5/reports/score_change_dynamic_threshold_metrics.json` | 2026-05-04 22:16:19 +07:00 |
| E6 | Score-change rule sweep summary | `docs/legacy/gbm-report-summaries/experiment4_score_change_rule_sweep/reports/score_change_rule_sweep_summary.csv` | 2026-05-04 22:15:01 +07:00 |
| E7 | Score-change rule sweep by event type | `docs/legacy/gbm-report-summaries/experiment4_score_change_rule_sweep/reports/score_change_rule_sweep_by_event_type.csv` | 2026-05-04 22:15:01 +07:00 |

## Current Research Framing

The current thesis direction is:

> Can a GBM-aware Transformer provide a statistically defensible financial anomaly detector by measuring inconsistency between learned window representations, reconstructed OHLCV dynamics, and GBM-implied return distributions, while remaining robust enough for early-warning and explanation-aware financial analysis?

The active model path is YAML-driven through root `run.py` and `main.py`. Stage code lives under the canonical `src/gbm/datasets.py`, `model.py`, `score.py`, `train.py`, `test.py`, `visualize.py`, and `statistics.py` surfaces.

## Research Questions

### RQ1. Is the constructed S&P 500 event-taxonomy dataset statistically meaningful and leakage-safe enough for model evaluation?

**Academic formulation.**  
Does the event-taxonomy labeling procedure produce anomaly labels that are statistically distinguishable from normal observations, while preserving a leakage-free time-based train/validation/test protocol?

**Current answer.**  
Answered at the dataset-validation level. E1 states that normalization leakage risk was removed because splitting occurs before normalization and the scaler is fit on training data only. E1 also reports strong distributional separation between anomaly and normal samples across `close_return`, `abs_return`, `volume_change`, `daily_range`, `rolling_vol_5`, and `rolling_vol_20`, with Mann-Whitney U and Kolmogorov-Smirnov p-values effectively zero.

**Interpretation.**  
The dataset appears suitable for anomaly-detection experimentation. The strongest empirical label signal is volatility-related rather than raw price direction. Regime analysis is also meaningful: anomaly rate is higher during the covid-crisis regime than during pre-covid or post-covid regimes.

**Limits.**  
This does not prove model superiority. It only supports the dataset as a defensible experimental substrate.

**Evidence.**  
E1, `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP1__insight.md`, dated 2026-04-13.

**Status.**  
Answered for dataset validity; still requires final reproducible validation report in the new repo.

### RQ2. Is a fixed power-law attention prior superior to a gaussian/local prior for financial anomaly detection?

**Academic formulation.**  
Does replacing the standard local gaussian attention prior with a heavier-tailed power-law prior produce robust improvements in anomaly detection, beyond merely changing the geometry of attention?

**Current answer.**  
Partially answered, and the answer is negative as a final design claim. E2 states that the power-law prior materially changes attention geometry: it increases prior entropy, reduces diagonal concentration, and promotes longer-range dependencies. However, detection-level gains are small, preprocessing-dependent, and not robust enough to justify a broad superiority claim.

**Interpretation.**  
The power-law prior is structurally meaningful but not sufficient as a standalone final architecture. The results motivate adaptive or regime-conditioned priors rather than a fixed prior-family choice.

**Limits.**  
The evidence does not include a publication-ready multi-seed significance test. Therefore, the defensible statement is that power-law attention changes representation geometry, not that it improves detection reliably.

**Evidence.**  
E2, `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md`, dated 2026-04-15.

**Status.**  
Answered enough to deprioritize fixed power-law as the final contribution.

### RQ3. Should the project shift from prior-discrepancy modeling to GBM-implied distribution consistency?

**Academic formulation.**  
Is a GBM-implied return-distribution consistency objective a more appropriate modeling target for daily equity anomaly detection than a fixed attention-prior discrepancy objective?

**Current answer.**  
Conceptually answered, experimentally incomplete. E3 defines the current GBM direction: the Transformer encodes each stock window, predicts a GBM-implied return distribution, compares it to the observed empirical return distribution, and combines distributional divergence with reconstruction error for anomaly scoring.

**Interpretation.**  
The shift is justified by E1 and E2 together. E1 shows anomaly signal is tied strongly to volatility and regime-sensitive market behavior. E2 shows fixed power-law priors alter representation but do not deliver robust detection improvement. The GBM formulation better aligns the anomaly score with financial return-distribution inconsistency.

**Limits.**  
This is a methodological direction, not yet a completed empirical answer. The new GBM model still needs benchmark, ablation, robustness, and statistical validation.

**Evidence.**  
E1, E2, E3.

**Status.**  
Direction selected; final empirical validation incomplete.

### RQ4. Does the current GBM score-change rule provide a useful operational anomaly alert?

**Academic formulation.**  
Can a dynamic threshold on absolute changes in GBM anomaly score identify anomaly windows with high precision while maintaining acceptable sensitivity?

**Current answer.**  
Partially answered. E5 and E6 show that the best observed rule in the curated sweep is rolling window 20, minimum periods 8, MAD multiplier 1.5. It produced:

```text
n_windows = 124502
tickers = 111
flagged_windows = 31009
precision = 0.9554
sensitivity = 0.2474
specificity = 0.7096
F1 = 0.3930
ROC-AUC from abs score change = 0.4578
PR-AUC from abs score change = 0.9559
TP = 29625
TN = 3382
FP = 1384
FN = 90111
```

The stricter rolling window 50, MAD multiplier 3.5 rule in E4 had much lower sensitivity:

```text
precision = 0.9514
sensitivity = 0.1185
specificity = 0.8479
F1 = 0.2107
flagged_windows = 14912
```

**Interpretation.**  
The score-change rule can produce very high precision, but sensitivity remains low. This is an operationally conservative alerting regime: when it flags, it is often correct, but it misses many anomaly windows.

**Limits.**  
The ROC-AUC from absolute score change is below 0.5 in the curated metrics, so the ranking quality of raw absolute score change is questionable. The high PR-AUC is influenced by the very high anomaly prevalence in the evaluated windows and should not be over-interpreted without class-balance analysis.

**Evidence.**  
E4, E5, E6.

**Status.**  
Partially answered. Useful as a high-precision diagnostic rule, not yet sufficient as a final early-warning system.

### RQ5. Which event types are captured by the current score-change rule?

**Academic formulation.**  
Does the dynamic score-change threshold detect specific event-taxonomy categories consistently, or does performance vary substantially across jump, drop, volume-spike, volatility-shock, and regime-shift events?

**Current answer.**  
Partially answered. For the rolling window 20, MAD k=1.5 rule, E7 reports:

```text
drop: catch_rate = 0.2366, events = 88183
jump: catch_rate = 0.2397, events = 88694
volatility_shock: catch_rate = 0.2476, events = 75929
volume_spike: catch_rate = 0.2455, events = 100768
regime_shift: events = 0 in the evaluated test windows
```

**Interpretation.**  
The rule catches roughly one quarter of jump, drop, volume-spike, and volatility-shock event windows under the best F1 configuration. There is no evidence for regime-shift detection in this evaluation because the event count is zero.

**Limits.**  
This does not support a strong event-specific superiority claim. It suggests broad but incomplete coverage across common event types.

**Evidence.**  
E7.

**Status.**  
Partially answered; event-specific claims remain weak.

### RQ6. Does the proposed GBM-aware Transformer outperform standard anomaly-detection baselines?

**Academic formulation.**  
Does the GBM-aware Anomaly Transformer significantly improve detection metrics over vanilla Anomaly Transformer, recurrent autoencoders, forecasting-based detectors, and classical statistical baselines?

**Current answer.**  
Not answered. The curated artifacts do not contain a complete baseline comparison for the current GBM formulation.

**Required evidence.**  
At minimum:

- Vanilla Anomaly Transformer baseline.
- LSTM or GRU autoencoder baseline.
- Forecasting-error baseline.
- Rolling volatility or z-score statistical baseline.
- Same train/validation/test split and tuning budget.
- Multiple seeds.
- Mean, standard deviation, confidence intervals, and paired tests.

**Status.**  
Open.

### RQ7. Does the model provide earlier warnings before anomaly onset?

**Academic formulation.**  
Does the proposed anomaly score detect events before their labeled onset, and how does its lead-time distribution compare against baseline detectors?

**Current answer.**  
Not answered. The score-change results report window-level detection and event catch rates, but they do not quantify lead time before event onset or detection delay after onset.

**Required evidence.**  
Lead-time metrics such as:

- Mean and median lead time.
- Distribution of lead times by event type.
- Fraction of events detected before onset.
- Detection delay for missed early warnings.
- Baseline comparison under identical event definitions.

**Status.**  
Open.

### RQ8. Is the current model robust across regimes, time periods, stock subsets, and random seeds?

**Academic formulation.**  
Does model performance remain stable under market-regime shifts, different stock universes, rolling evaluation windows, and repeated random seeds?

**Current answer.**  
Not answered for the current GBM model. E1 supports that labels are regime-sensitive, but it does not prove model robustness. E2 discusses regime-level diagnostics for the prior study, but not as a final GBM robustness result.

**Required evidence.**  
Regime-split and multi-seed evaluation:

- Pre-crisis, crisis, post-crisis or equivalent regime buckets.
- Multiple random seeds.
- Stock-subset generalization.
- Rolling-window backtest.
- Confidence intervals and significance tests.

**Status.**  
Open.

### RQ9. Are the model explanations faithful, stable, and useful?

**Academic formulation.**  
Can the model produce explanations whose attributed features, windows, or stock relations causally affect anomaly scores and remain stable under reasonable perturbations?

**Current answer.**  
Not answered. Current artifacts do not include deletion/insertion, explanation stability, or faithfulness evaluation for the GBM model.

**Required evidence.**  
At minimum:

- Deletion test: remove top-attributed features and verify anomaly score changes appropriately.
- Insertion test: restore attributed features and verify score recovery.
- Stability under noise, seed variation, and nearby windows.
- Clear distinction between plausible visualization and faithful explanation.

**Status.**  
Open.

### RQ10. Do cross-sectional stock relations add measurable value?

**Academic formulation.**  
Does incorporating cross-sectional context among related equities improve detection, early warning, or explanation quality beyond independent per-ticker temporal modeling?

**Current answer.**  
Not answered. The active GBM path is a pooled multi-ticker window pipeline, but the curated evidence does not demonstrate an explicit cross-sectional relation module or an ablation isolating its contribution.

**Required evidence.**  
Ablations comparing:

- Per-ticker temporal-only model.
- Pooled multi-ticker model without relation modeling.
- Explicit cross-sectional relation model.
- Sector or correlation-informed relation variants.

**Status.**  
Open.

### RQ11. Which parts of the legacy research direction should no longer be treated as active?

**Academic formulation.**  
Which hypotheses or implementation paths have been superseded by the current GBM-aware formulation and should be retained only as historical context?

**Current answer.**  
Answered operationally. The fixed power-law prior path should no longer be treated as the central contribution. E2 shows it is structurally meaningful but not empirically strong enough as a standalone prior. The active direction is GBM distribution consistency plus dynamic score-change evaluation.

**Deprecated as primary direction.**

- Fixed gaussian-vs-power-law prior as the final architecture.
- Broad old Anomaly Transformer code path.
- Old Windows-only runner assumptions.
- Bulk old outputs as active evidence.

**Retained as context.**

- Dataset validation evidence.
- Power-law diagnostic insight.
- Motivation for adaptive priors.
- Event taxonomy and regime-sensitivity findings.

**Evidence.**  
E1, E2, E3, `docs/AGENT_CONTEXT_AUDIT.md`.

**Status.**  
Answered for project organization.

## Current Overall Answer

The project has answered enough to justify the current methodological shift: the event-taxonomy dataset appears statistically meaningful and leakage-aware; fixed power-law attention is not sufficiently robust as the final contribution; and GBM-implied distribution consistency is a defensible next modeling direction. The score-change dynamic-threshold results show high precision but low sensitivity, so they are useful diagnostically but not yet sufficient for a final early-warning claim.

The project has not yet answered the main publication questions: whether the GBM-aware Transformer beats fair baselines, whether it provides earlier warning, whether it generalizes across regimes and stocks, and whether explanations are faithful.

## Recommended Next Experiments

1. Re-run the active GBM joint pipeline in the new repo and store a small curated report under `docs/reports/`.
2. Add baseline detectors under the same split and evaluation protocol.
3. Run multi-seed experiments and report confidence intervals.
4. Add lead-time and detection-delay evaluation.
5. Add regime-split and stock-subset robustness tables.
6. Add explanation faithfulness only after the explanation mechanism is implemented.
7. Promote only statistically supported claims to paper text.
