# EXP2 Power Prior Diagnostic Checklist

This checklist is the execution companion to [TODO/EXP2/experiment_plan.md](TODO/EXP2/experiment_plan.md). The plan defines the scripts, outputs, and metrics; this file is the run-time checklist.

Use this checklist to decide whether the power-law prior is still worth keeping, or whether it should be replaced by a more flexible prior design.

## 0. Decision Goal
- [ ] Decide whether the current power-law assumption is:
  - [ ] useful as-is,
  - [ ] useful only for some event types or regimes,
  - [ ] too rigid and should be replaced with a more flexible prior,
  - [ ] not useful enough to keep in the benchmark.
- [ ] Do not judge the prior only by ROC-AUC or reconstruction loss.
- [ ] Do not conclude failure or success from a single ticker.
- [ ] Use the same time split, feature set, and seed list for all diagnostic variants.

## 1. Hypothesis Definition
- [ ] State the hypothesis before running any diagnostic.
- [ ] Hypothesis A: power-law prior should help long-memory dynamics, but may underfit short event-driven anomalies.
- [ ] Hypothesis B: the current power-law implementation may be too weak because it only affects attention prior, not the main loss directly.
- [ ] Hypothesis C: distance clamping may reduce contrast between near and far positions.
- [ ] Hypothesis D: batch normalization and validation-based thresholding may reduce visible differences between priors.
- [ ] Record the exact version of code used for the run.

## 2. Experimental Variants
- [ ] Run gaussian prior baseline.
- [ ] Run power-law prior baseline.
- [ ] Run both variants with `normalize-batch` on.
- [ ] Run both variants with `normalize-batch` off.
- [ ] Keep all other hyperparameters fixed.
- [ ] If possible, repeat each setting with multiple seeds.
- [ ] If possible, keep the same ticker set as Experiment 1 for direct comparison.

## 3. Distribution Diagnostics for Learned Prior Parameters
- [ ] Save learned `alpha` for power-law runs.
- [ ] Save learned `sigma` for gaussian runs.
- [ ] Compare `alpha` and `sigma` distributions for normal windows vs anomaly windows.
- [ ] Compare `alpha` and `sigma` distributions per ticker.
- [ ] Compare `alpha` and `sigma` distributions per regime.
- [ ] Check whether `alpha` collapses toward values that make the prior effectively uniform.
- [ ] Check whether `alpha` becomes extremely large, which would make the prior overly local.
- [ ] Check whether `alpha` is highly unstable across seeds or windows.
- [ ] Report mean, standard deviation, and confidence interval for these parameter summaries.

## 4. Attention and Prior Behavior Checks
- [ ] Compute attention/prior divergence, not only detection metrics.
- [ ] Measure KL divergence or another divergence between learned attention and the prior.
- [ ] Measure how divergence changes on normal vs anomaly windows.
- [ ] Measure whether divergence changes by event class.
- [ ] Visualize attention maps for representative normal and anomaly windows.
- [ ] Check whether the power-law prior actually changes attention, or whether the learned attention mostly ignores it.

## 5. Event Taxonomy Slice Analysis
- [ ] Split results by event type.
- [ ] Report results separately for `jump`.
- [ ] Report results separately for `drop`.
- [ ] Report results separately for `volume_spike`.
- [ ] Report results separately for `volatility_shock`.
- [ ] Report results separately for `regime_shift`.
- [ ] Check whether power-law helps only on long-duration or high-volatility classes.
- [ ] Check whether power-law hurts short, sharp event types.
- [ ] Do not average away class-specific failures.

## 6. Normalization Sensitivity
- [ ] Compare gaussian vs power-law with batch normalization enabled.
- [ ] Compare gaussian vs power-law with batch normalization disabled.
- [ ] Check whether normalization reduces amplitude differences that may matter for jump/drop detection.
- [ ] Check whether normalization weakens long-range structure in the attention prior.
- [ ] Check whether any observed benefit disappears only after normalization, which would indicate a preprocessing interaction rather than a true model gain.

## 7. Threshold Sensitivity
- [ ] Verify that threshold selection uses validation only.
- [ ] Compare the current 3-sigma threshold with at least one alternative thresholding rule.
- [ ] Check whether gaussian and power-law need different thresholds to be fairly compared.
- [ ] Measure sensitivity of precision, recall, and early warning behavior to threshold choice.
- [ ] Check whether the prior affects anomaly score ranking even when thresholding is held fixed.

## 8. Long-Memory vs Event-Driven Test
- [ ] Define a proxy for long-memory windows.
- [ ] Define a proxy for short event-driven windows.
- [ ] Compare power-law and gaussian performance on both groups.
- [ ] Check whether power-law is better only when the window contains persistent stress, not discrete shocks.
- [ ] Check whether the prior helps in regime-like behavior more than in point anomaly behavior.

## 9. Interpretation of Failure Modes
- [ ] If power-law is worse on short events but better on long events, mark it as regime- or duration-specific rather than globally bad.
- [ ] If alpha collapses toward uniform behavior, treat the prior as underused.
- [ ] If alpha becomes overly local, treat the prior as too sharp for this dataset.
- [ ] If the prior effect disappears after normalization, treat preprocessing as a likely confounder.
- [ ] If the attention/prior divergence stays small but metrics do not improve, the prior may be too weak to matter.

## 10. Candidate Modifications If Power-Law Is Partially Useful
- [ ] Try a regime-conditioned prior that changes with volatility or market state.
- [ ] Try a piecewise prior with local decay at short range and heavy-tail decay at long range.
- [ ] Try a feature-specific prior that separates OHLCV and returns.
- [ ] Try an event-type-aware prior if class-specific differences are strong.
- [ ] Try a mixture prior that interpolates between gaussian and power-law with a learned gate.
- [ ] Try a stronger coupling between prior and training objective if the current prior influence is too weak.

## 11. Required Outputs
- [ ] Table of learned `alpha` and `sigma` summaries.
- [ ] Table of attention/prior divergence by ticker and event type.
- [ ] Table of gaussian vs power-law results with and without batch normalization.
- [ ] Table of event-type-sliced results.
- [ ] Table of threshold sensitivity results.
- [ ] Figure showing representative attention maps for normal and anomaly windows.
- [ ] Figure or table showing whether power-law collapses to uniform, local, or noisy behavior.

## 12. Go / No-Go Rule
- [ ] Go: keep power-law only if it gives a consistent, statistically defensible gain on at least one meaningful slice, and the gain survives threshold and normalization checks.
- [ ] Partial go: keep it only as a regime-specific or event-specific variant if it is not globally better.
- [ ] No-go: replace it if it is consistently indistinguishable from gaussian, or if it is unstable and its apparent effect vanishes after diagnostics.

## 13. Reporting Rule
- [ ] If the result is not significant, write it as a trend.
- [ ] If the prior helps only on some event types, say so explicitly.
- [ ] If the prior does not help after diagnostics, do not claim that the power-law assumption is validated.
- [ ] Do not use a qualitative example to override aggregate statistics.
