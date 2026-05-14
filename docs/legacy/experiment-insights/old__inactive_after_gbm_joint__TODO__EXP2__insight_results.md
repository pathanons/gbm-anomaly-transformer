# EXP2 Insight Results: Power-Law Prior Diagnostic Study

## Summary of the Main Finding
The EXP2 diagnostic sweep shows that the power-law prior materially changes the inductive bias of the Anomaly Transformer, but the downstream detection gain remains modest and is not yet strong enough to support a broad superiority claim. In particular, the power-law prior behaves as intended from a modeling perspective: it redistributes attention mass away from the local diagonal and toward longer-range dependencies. However, this shift does not translate into a large or consistently significant improvement in anomaly detection across the current evaluation setup.

## What the Diagnostics Show
The attention summaries indicate a clear structural difference between gaussian and power-law priors. On the test split with batch-wise normalization, the gaussian prior remains highly local, with low prior entropy and strong diagonal concentration. By contrast, the power-law prior has much higher prior entropy and substantially lower diagonal mass, which is consistent with a flatter, heavier-tailed dependency structure. In other words, the power-law implementation is not degenerate; it is genuinely encoding long-range dependence.

This behavior supports the hypothesis that the two priors are not equivalent in representation space. The power-law variant is not simply a noisy reparameterization of gaussian attention. Instead, it learns a qualitatively different prior geometry, with greater mass assigned to distant positions and greater effective support across the window.

## Detection-Level Interpretation
At the detection level, the gains are small and preprocessing-dependent. With `normalize-batch` enabled, the power-law model achieves a slightly higher test ROC-AUC than gaussian and also a marginally higher PR-AUC, but the absolute differences remain limited. The recall remains low for both priors, while precision is extremely high because the validation-derived threshold is conservative. This implies that the current operating point favors specificity over sensitivity, and the observed improvement is closer to a weak trend than a decisive result.

When batch normalization is disabled, the difference between priors becomes even smaller. In that setting, gaussian and power-law are effectively tied on ROC-AUC, with only negligible changes in PR-AUC and F1. This suggests that any benefit from the power-law assumption is not robust to preprocessing choice and may depend on the normalization regime rather than on the prior family alone.

## Interpretation by Hypothesis
The results are consistent with the original concern that financial anomalies are often event-driven and short-horizon, while the power-law assumption is more natural for long-memory dynamics. The diagnostics indicate that power-law does indeed promote long-range structure, but the benchmark task is not dominated by that structure alone. In short, the assumption is plausible, but not yet sufficiently aligned with the dominant anomaly signatures in this dataset to produce a large benchmark gain.

The second concern is also supported: because the power-law prior only enters through the attention prior term, its effect is likely weaker than the reconstruction objective that drives optimization. The model can therefore learn a long-range prior without that prior necessarily dominating the reconstruction-error ranking used for anomaly scoring.

The third and fourth concerns remain relevant. The validation-based thresholding produces a conservative classifier, which compresses the difference between priors at the final decision level. As a result, the prior can change the internal representation substantially while still producing only small external metric changes. This is a classic case where representation-level change does not immediately imply decision-level gain.

## Event-Slice and Regime-Level Reading
The event-slice outputs do not yet show a clean, publication-ready pattern that would justify a strong claim about one event class benefiting more than another. The slice tables are useful diagnostically, but they are not yet stable enough to support a narrow event-specific claim without additional statistical testing and clearer aggregation.

Regime coverage is present, and the diagnostics pipeline is capable of separating pre-crisis, crisis, and post-crisis windows. That said, the current EXP2 run is better interpreted as evidence that the prior family interacts with regime structure, not as evidence that the interaction is already strong enough to anchor a final claim.

## Practical Conclusion
The most defensible conclusion from EXP2 is the following:

1. The power-law prior is structurally meaningful and does alter the model’s attention geometry.
2. The current benchmark does not show a large, robust, or clearly significant improvement over gaussian.
3. The effect appears sensitive to batch normalization and the current thresholding strategy.
4. The power-law assumption should therefore be treated as a candidate prior family, not as a validated final design.

## Research Implication
The next step is not to discard the idea outright, but to make the prior more adaptive. The diagnostics motivate a follow-up design in which the prior is conditioned on regime or event type, or blended with a local component through a learned gate. This would preserve the long-memory inductive bias while allowing the model to remain sensitive to short, discrete financial shocks.

## Paper-Ready Wording
The diagnostic study indicates that the proposed power-law prior induces a substantially heavier-tailed attention structure than the gaussian prior, as reflected by higher prior entropy and lower diagonal concentration. However, this representational change produces only modest and preprocessing-dependent improvements in anomaly detection. With batch-wise normalization enabled, the power-law model shows a slight advantage in ROC-AUC and PR-AUC, but the effect is small and the recall remains low. Without batch-wise normalization, the difference between priors becomes negligible. Overall, the evidence suggests that the power-law assumption is structurally valid but not yet sufficient as a standalone prior for robust financial anomaly detection, motivating more flexible regime-aware or event-aware prior formulations.

## Next Architecture Proposal
The next version of Anomaly Transformer should not be a single-prior model. It should be a modular financial anomaly framework that preserves the backbone idea of attention-based dependency modeling while adding explicit market structure, regime sensitivity, and detection calibration.

### Design Principles From Validation and EXP2
- The dataset contains real anomaly signal, especially in volatility-related behavior, so the model should keep a strong temporal shock-detection path.
- Anomaly incidence changes materially by regime, so the model should not assume one dependency geometry for all periods.
- Batch-wise normalization can attenuate event amplitude, so the model should be robust to preprocessing and not depend on one scaling choice.
- Power-law attention is useful as an inductive bias, but not sufficient by itself, so the prior should become adaptive rather than fixed.
- Since regime_shift is sparse, the model should treat regime awareness as a conditioning mechanism, not as a hard rule baked into every window.

### Proposed Modular Architecture

| Block | Purpose | What It Should Learn |
| --- | --- | --- |
| Input feature stem | Encode OHLCV and returns before attention | Preserve raw event amplitude while reducing scale noise |
| Shock encoder | Capture short-horizon jumps, drops, and volume spikes | Local burst structure and asymmetric moves |
| Multi-scale temporal encoder | Model short, medium, and longer lag structure | Collective anomalies and persistent stress windows |
| Regime encoder | Summarize market state from rolling context | Calm vs stressed vs crisis-like behavior |
| Cross-sectional context encoder | Capture stock-to-stock and sector dependencies | Spillover, leader-follower, and market-wide effects |
| Adaptive prior controller | Select or blend prior families by context | When to behave locally, when to behave long-range |
| Detection head | Produce anomaly score | Final ranking for alerting and thresholding |
| Event-type head | Predict jump/drop/volume_spike/volatility_shock/regime_shift | Separate anomaly detection from event semantics |
| Explanation head | Attribute score to features and relations | Produce faithful feature and peer explanations |
| Calibration layer | Map raw scores to stable alerts | Control precision-recall trade-off and lead-time behavior |

### How The Blocks Should Interact
1. The input stem first projects OHLCV and return features into a shared latent space.
2. The shock encoder focuses on local volatility and return discontinuities, because these are the strongest signals observed in validation.
3. The multi-scale temporal encoder expands the receptive field so the model can recognize both isolated shocks and multi-day collective stress.
4. The regime encoder provides a context vector that conditions the prior and the scoring head.
5. The cross-sectional encoder injects market structure so the model can distinguish idiosyncratic anomalies from sector or market spillovers.
6. The adaptive prior controller blends a local component and a heavy-tailed component instead of choosing one fixed geometry.
7. The detection head computes the anomaly score, while the event-type head optionally helps the representation separate different financial event families.
8. The explanation head reports which features or relations contributed to the alert, but only if faithfulness checks remain stable.
9. The calibration layer converts raw anomaly evidence into operational alerts using validation-only thresholds.

### Practical Prior Design
The fixed gaussian versus power-law choice should be replaced by a gated mixture:
- local prior for sharp shocks and short event windows
- heavy-tailed prior for long-memory and persistence
- regime-conditioned mixing weights so crisis periods can use a different balance from calm periods

This preserves the original intuition behind gaussian attention, but removes the assumption that one dependency shape is valid everywhere.

### Why This Is Better For Stocks
Stocks do not behave like a single stationary sequence. They show local shocks, regime changes, and cross-sectional contagion. A good anomaly model for this domain should therefore be able to:
- detect abrupt local deviations
- adapt when the market regime changes
- borrow context from related stocks
- keep alerts calibrated under different preprocessing choices

That is the main modification direction implied by EXP1 and EXP2.

## Research Direction Summary
The strongest next-step claim is not that power-law is the final answer, but that the market requires a context-adaptive anomaly model. The empirical evidence supports a framework in which Anomaly Transformer remains the backbone, while the prior, scoring, and explanation mechanisms become conditional on regime, event type, and cross-sectional context. In paper terms, the contribution becomes a detector plus early-warning system plus explanation-aware financial anomaly framework, rather than a single-prior attention model.
