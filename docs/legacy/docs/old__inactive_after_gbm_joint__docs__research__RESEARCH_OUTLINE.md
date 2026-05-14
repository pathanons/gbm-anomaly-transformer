# Research Outline

## Working Title
Interpretable Hybrid Anomaly Detection for U.S. Equity Markets Using an Enhanced Anomaly Transformer

## 1. Research Goal
Develop and validate a hybrid anomaly detection framework for daily U.S. large-cap equities that is:
- statistically defensible,
- early-warning oriented,
- robust across market regimes,
- and capable of producing faithful explanations.

## 2. Core Research Questions
1. Does the proposed model improve early warning lead time for financial anomalies?
2. Does it improve detection quality over standard and classical baselines?
3. Do the added components, such as multi-scale temporal modeling and cross-sectional stock relations, provide measurable gains?
4. Are the explanations faithful, stable, and useful enough to support a publication claim?
5. Does the method remain robust across time regimes, stock subsets, and random seeds?

## 3. Non-Negotiable Evaluation Principles
- If there is no significance test or confidence interval, do not write that the model is better.
- If the difference is small and not significant, state that the result is only a trend.
- If an explanation looks plausible but fails faithfulness checks, do not claim interpretability.
- If early warning improves but precision drops, report the trade-off clearly.
- Validation data only may be used for thresholding, model selection, and tuning.
- Test data must remain untouched until final evaluation.

## 4. Experiment Plan

### Experiment 1: Main Benchmark
Goal: Compare the proposed model against the main baselines.

Baselines:
- Vanilla Anomaly Transformer
- LSTM autoencoder
- GRU autoencoder
- Forecasting-based detectors
- Simple statistical baseline such as z-score or rolling volatility rule

Metrics:
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- event-level false alarm rate

Statistics:
- multiple seeds
- mean and standard deviation
- 95% confidence interval
- paired significance test against the strongest baseline

### Experiment 2: Ablation Study
Goal: Isolate each proposed contribution.

Ablations:
- remove multi-scale temporal module
- remove cross-sectional relation module
- remove explanation head
- use temporal-only versus temporal plus cross-sectional modeling
- use detection-only versus detection plus explanation

Metrics:
- same detection metrics as Experiment 1
- early warning metrics
- explanation metrics where applicable

### Experiment 3: Early Warning Study
Goal: Quantify whether the model alerts earlier.

Metrics:
- lead time before anomaly onset
- detection delay
- fraction of events detected before onset
- early warning recall

Statistics:
- report distribution, not only average
- use paired comparison across events or windows

### Experiment 4: Regime Robustness
Goal: Check whether performance holds across market conditions.

Regimes:
- bullish period
- bearish period
- high-volatility period
- low-volatility period
- pre-crisis, crisis, post-crisis splits if available

Metrics:
- detection metrics per regime
- early warning metrics per regime
- variance across regimes

### Experiment 5: Explanation Faithfulness
Goal: Test whether explanation output is causally useful.

Checks:
- deletion test: remove top-attributed features and verify score drops
- insertion test: restore top-attributed features and verify score rises
- fidelity against rule-based labels or event descriptors
- overlap with known event drivers when available

Rule:
- if faithfulness fails, explanation claims must be reduced or removed.

### Experiment 6: Stability and Generalization
Goal: Test whether results are stable and transferable.

Checks:
- run multiple random seeds
- evaluate across rolling windows
- compare seen versus unseen stock subsets
- compare different time spans or market segments

Metrics:
- variance across seeds
- confidence interval width
- performance drift over time
- explanation stability

### Experiment 7: Qualitative Case Study
Goal: Provide supporting examples, not primary evidence.

Rules:
- perform only after the main statistical results are finalized
- use it to illustrate examples, not to justify claims
- do not use case studies to hide weak aggregate results

## 5. Recommended Paper Order
1. Main benchmark
2. Ablation study
3. Early warning study
4. Regime robustness
5. Explanation faithfulness
6. Stability and generalization
7. Qualitative case study

## 6. Statistical Reporting Standard
Every table or figure must follow this standard:
- report mean and standard deviation when repeated runs exist
- report confidence intervals when feasible
- use hypothesis tests for baseline comparisons
- report effect size where appropriate
- correct for multiple comparisons when many baselines are tested

Preferred wording rules:
- say "better" only when supported by significance and confidence intervals
- say "trending higher" or "trending lower" when the evidence is suggestive but not decisive
- say "not significantly different" when the test does not support a clear difference

## 7. Publication Positioning
The final paper should argue that the model is:
- a detector,
- an early warning system,
- and an explanation-aware financial anomaly framework.

The publication target should remain credible for IEEE-style AI/ML venues while staying grounded in financial analysis.
