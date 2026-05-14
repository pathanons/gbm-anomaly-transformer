# EXP1 Insights

## Core Takeaways
- The SP500 event-taxonomy dataset is no longer showing normalization leakage risk in the loader. Split now happens before normalization, and the scaler is fit on train only, so validation and test are transformed without using future information.
- The generated labels are not random noise. Statistical tests show strong separation between anomaly and normal samples across multiple features, with Mann-Whitney U and KS p-values effectively zero for close_return, abs_return, volume_change, daily_range, rolling_vol_5, and rolling_vol_20.
- The most discriminative signals are volatility-based, not raw price direction. rolling_vol_5 and rolling_vol_20 show the strongest separation, which is consistent with financial anomalies being volatility events more than simple level shifts.
- The regime analysis is meaningful. Anomaly rate is much higher in the covid_crisis regime than in pre_covid or post_covid, so the labels capture time-varying market stress rather than a flat background rate.
- The label taxonomy is empirically usable, but not all classes are equally populated. regime_shift remains very sparse, so it is not yet a strong standalone class for downstream benchmarking.
- The pipeline now fails loudly when artifacts are missing, which makes the experiment easier to trust because silent failures are less likely to contaminate reported results.

## Statistical Interpretation
- The very small p-values mean the null hypothesis of no difference between anomaly and normal groups is rejected for the tested features.
- Effect size still matters. Some features are strongly separated in practice, while close_return is statistically significant but weak in magnitude.
- The chi-square result for regime vs anomaly supports the claim that anomaly incidence is regime-dependent, not uniformly distributed through time.
- Bootstrap-based reporting is important because the dataset is large enough that p-values alone would overstate confidence if effect sizes were ignored.

## Research Implications
- The dataset is suitable for publication-grade anomaly detection experiments because it has time-ordered structure, statistically visible signal, and evidence of regime sensitivity.
- Benchmark claims should focus on volatility-aware detection and regime robustness rather than only pointwise return prediction.
- Interpretation claims should be made carefully. The current evidence supports separability of anomaly labels, but not explanation faithfulness yet.
- Rolling-backtest validation and leakage checks are now part of the evidence chain, which strengthens the experimental protocol for IEEE-style reporting.

## Remaining Caveats
- Statistical significance does not guarantee model usefulness; benchmark comparisons still need multi-seed evaluation and fair baselines.
- Sparse classes such as regime_shift may need either revised thresholds or a different experimental framing if they are to be used as a major result.
- The current validation supports dataset quality and signal presence, but not final claims about superiority of any model.

## Thesis-Ready Text
The SP500 event-taxonomy dataset exhibits statistically meaningful separation between anomaly and normal observations across multiple market features. In particular, Mann-Whitney U tests and Kolmogorov-Smirnov tests indicate that close_return, abs_return, volume_change, daily_range, rolling_vol_5, and rolling_vol_20 differ strongly between the two groups, with p-values effectively zero in the current validation run. Among these variables, volatility-based features provide the clearest separation, suggesting that the taxonomy captures anomaly behavior more as volatility shocks than as simple directional price movements.

The regime analysis further shows that anomaly incidence is not uniform over time. The covid_crisis period contains a substantially higher anomaly rate than the pre_covid and post_covid regimes, and the chi-square test confirms that anomaly occurrence is associated with market regime. This indicates that the constructed labels preserve economically meaningful temporal structure rather than producing a flat background anomaly rate. At the preprocessing level, the loader has been refactored so that data splitting occurs before normalization, and the normalization scaler is fit on the training set only. As a result, the current validation reports no normalization leakage risk from source inspection. Taken together, these results support the dataset as a defensible basis for publication-grade anomaly detection experiments, while still requiring fair baseline comparison, multi-seed evaluation, and robustness analysis before any final superiority claims can be made.