# Research Plan: Loss, Shift, and Endpoint Scoring

This is the active working plan for the GBM anomaly-detection research track. Treat it as the source of truth for the current experimental roadmap and update it after implementation runs with concise results, failures, and decisions.

## Scope

The current project uses the pooled GBM joint pipeline. The plan is ordered so that each phase builds on the previous one:

1. Loss refactor
2. Distribution-shift score study
3. Association discrepancy study
4. Multi-channel anomaly score
5. Multi-channel evaluation and reporting
6. Endpoint score, thresholding, and regime separation
7. Baseline benchmarking

All thresholds, hyperparameters, and score-selection choices must use validation data only.

## Phase 1: Loss Refactor With Explicit Terms

1. Confirm the current control points in `src/gbm/losses.py`, `src/gbm/scoring.py`, `scripts/gbm/validate_joint.py`, and `scripts/gbm/test_joint.py` so training loss, diagnostics, distribution-shift score, and final anomaly score are not conflated.
2. The current scalar loss is:

$$
L_{old} = w_{nll} L_{nll} + w_{rec} L_{rec} + w_{div} L_{div} + w_{assoc} L_{assoc}
$$

where `L_div` is currently moment-based matching and `L_assoc` is symmetric KL between attention and prior.
3. Replace `L_div` with the normalized two-term formulation:

$$
L_{div}^* = L_{mean} + L_{vol}
$$

with

$$
L_{mean} = \frac{(\mu_{obs} - \mu_{pred})^2}{\sigma_{pred}^2 / T + \epsilon}
$$

$$
L_{vol} = \left[\log\left(\frac{\sigma_{obs} + \epsilon}{\sigma_{pred} + \epsilon}\right)\right]^2
$$

where `T` is the window length, `mu_obs` and `sigma_obs` come from observed returns, and `mu_pred`, `sigma_pred` come from the predictive head.
4. Keep `L_nll` as the main predictive term, `L_rec` as an auxiliary reconstruction term, and use `L_assoc` as a diagnostic or regularizer rather than part of the final score in this phase.
5. Return training/evaluation components separately:

$$
S_{point} = -\log p(r_t \mid x_{t-T:t-1})
$$

$$
S_{recon} = \frac{1}{F} \sum_{f=1}^{F} (\hat{x}_{t,f} - x_{t,f})^2
$$

$$
S_{dist} = L_{div}^*
$$

$$
S_{assoc} = \frac{1}{LH} \sum_{l,h} SKL(A^{(l,h)} \Vert P^{(l,h)})
$$

6. Select any weights through validation data only, and report sensitivity of `L_div*` and `L_assoc` weights before declaring stability.

## Phase 2: Distribution-Shift Score Study

7. Define the current baseline as the existing moment-based scalar score that aggregates `mu_obs` and `sigma_obs` over the full window.
8. Study at least three candidate distribution-shift metrics:

### Quantile Wasserstein

$$
L_{QW2} = \frac{1}{K} \sum_{k=1}^{K} \left[ Q_{obs}(u_k) - Q_{pred}(u_k) \right]^2
$$

with

$$
u_k = \frac{k - 0.5}{K}
$$

### Tail-weighted Quantile Wasserstein

$$
L_{QW2\_tail} = \frac{\sum_k \omega(u_k) \left[ Q_{obs}(u_k) - Q_{pred}(u_k) \right]^2}{\sum_k \omega(u_k)}
$$

with

$$
\omega(u) = 1 + \gamma \lvert u - 1/2 \rvert^p
$$

### Energy Distance

$$
L_{energy} = \frac{2}{TM} \sum_{i=1}^{T} \sum_{j=1}^{M} \lvert r_i - \tilde{r}_j \rvert - \frac{1}{T^2} \sum_{i,j} \lvert r_i - r_j \rvert - \frac{1}{M^2} \sum_{i,j} \lvert \tilde{r}_i - \tilde{r}_j \rvert
$$

### MMD

$$
MMD^2(P,Q) = \frac{1}{T^2} \sum_{i,j} k(r_i,r_j) + \frac{1}{M^2} \sum_{i,j} k(\tilde{r}_i,\tilde{r}_j) - \frac{2}{TM} \sum_{i,j} k(r_i,\tilde{r}_j)
$$

9. Keep `K`, kernel choice, sample budget, and quantile count fixed per experiment.
10. Validate whether the proposed score separates tail shift and regime shift better than the baseline, and whether false alarms or detection delay improve.
11. Use validation-only calibration for variant selection, thresholds, and hyperparameters.
12. Store raw, calibrated, and derived statistics separately.

## Phase 3: Association Discrepancy Study

13. Baseline association discrepancy is the current mean symmetric KL across layers, heads, and positions:

$$
S_{assoc\_old} = \frac{1}{LHT} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{i=1}^{T} SKL(A_{l,h,i,.} \Vert P_{l,h,i,.})
$$

14. Study endpoint association score:

$$
S_{assoc\_end} = \frac{1}{LH} \sum_{l=1}^{L} \sum_{h=1}^{H} SKL(A_{l,h,T,.} \Vert P_{l,h,T,.})
$$

15. Study tail-window association score:

$$
S_{assoc\_tail}(Q) = \frac{1}{LHQ} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{i=T-Q+1}^{T} SKL(A_{l,h,i,.} \Vert P_{l,h,i,.})
$$

with at least `Q = 5` and `Q = 10`.
16. Compare whether endpoint or tail-window variants improve localization, reduce averaging dilution, and reduce overlap effects.
17. Store `S_assoc_old`, `S_assoc_end`, `S_assoc_tail(Q)`, and validation/test statistics separately.

## Phase 4: Multi-Channel Anomaly Score

18. Define the anomaly score as a vector first, not a scalar:

$$
S_t = [S_{point,t}, S_{dist,t}, S_{recon,t}, S_{assoc,t}, S_{jump,t}]
$$

19. Define each channel explicitly:

$$
S_{point,t} = -\log p(r_t \mid x_{t-T:t-1})
$$

$$
S_{dist,t} = L_{QW2\_tail}
$$

$$
S_{recon,t} = \frac{1}{F} \sum_{f=1}^{F} (\hat{x}_{t,f} - x_{t,f})^2
$$

$$
S_{assoc,t} = S_{assoc\_end} \; \text{or} \; S_{assoc\_tail}(Q)
$$

$$
S_{jump,t} = \lvert S_{dist,t} - S_{dist,t-1} \rvert
$$

20. Normalize each channel with validation statistics:

$$
Z_{k,t} = \frac{S_{k,t} - median_{train}(S_k)}{1.4826 \cdot MAD_{train}(S_k) + \epsilon}
$$

for `k ∈ {point, dist, recon, assoc, jump}`.
21. When a scalar score is needed, combine the normalized components as:

$$
S_{final,t} = \alpha Z_{point,t} + \beta Z_{dist,t} + \gamma Z_{recon,t} + \delta Z_{assoc,t} + \eta Z_{jump,t}
$$

22. Use validation data only to fit and test the weights `α, β, γ, δ, η`.
23. Keep vector components, normalized components, and scalar aggregate outputs separate.

## Phase 5: Multi-Channel Evaluation And Reporting

24. Evaluate both component-level and scalar-level metrics, including `AUCPR`, `ROC-AUC`, `event-level recall`, `detection delay`, `false alarms per year`, and correlations between score components.
25. Run ablations at least for `point-only`, `dist-only`, `recon-only`, `assoc-only`, `jump-only`, `point+dist`, `point+dist+recon`, and the full vector.
26. Report validation sensitivity across several weight settings before choosing a final scalar combination.
27. Keep this phase separate from endpoint scoring so component behavior can be understood before changing inference granularity.

## Phase 6: Endpoint Score, Thresholding, And Regime Separation

28. After the loss, distribution-shift, association, and multi-channel phases are stable, move to per-day inference.
29. Aggregate endpoint scores robustly over overlapping windows:

$$
S_{k,t}^{ep} = median_{w \in W_t} S_{k,w}
$$

where `W_t` is the set of windows covering day `t`.
30. For point anomalies, use direct endpoint predictive score:

$$
S_{point,t}^{ep} = -\log p(r_t \mid x_{t-T:t-1})
$$

31. Use rolling robust thresholding:

$$
\tau_t = median(S_{t-w:t-1}) + c \cdot 1.4826 \cdot MAD(S_{t-w:t-1})
$$

and flag anomalies when `S_t^{ep} > τ_t`.
32. If needed, add EVT/POT:
    - choose a threshold `u`
    - define exceedances `Y = S - u | S > u`
    - fit a Generalized Pareto Distribution to `Y`
33. Separate point anomaly and regime shift with change-point detection on `S_{dist,t}^{ep}` and volatility.
    - consider CUSUM:

$$
G_t = max(0, G_{t-1} + (S_t^{ep} - \nu))
$$

    - consider PELT for multiple changepoints.
34. Report whether endpoint scoring improves localization, early alerting, false-alarm control, and regime separation.

## Phase 7: Baselines And Final Benchmarking

35. Expand the baseline suite to cover:
    - statistical volatility baselines: rolling volatility, EWMA volatility, ARCH, GARCH, GARCH standardized residual
    - tail-statistics baselines: standardized residual, rolling quantile, EVT/POT, EVT on GARCH-filtered residuals
    - classical anomaly detection: Isolation Forest, One-Class SVM, Local Outlier Factor, robust covariance-based methods
    - deep temporal anomaly detection: LSTM Encoder-Decoder, Transformer Autoencoder, TranAD, vanilla Anomaly Transformer
36. Tune baselines only on validation data and under a comparable budget.
37. Benchmark with `AUCPR`, `ROC-AUC`, `event-level recall`, `detection delay`, `false alarms per year`, calibration curves, and statistical tests.
38. Use the existing baseline scripts in `scripts/gbm/` as the primary execution path.

## Reporting Rules

- Always report mean, standard deviation, and seed-level variation when multiple runs are available.
- Do not use test data to select thresholds, weights, or model variants.
- Keep the active GBM joint pipeline as the reference execution path.
- Update this file after each implementation batch with a short note of what changed, what failed, and what still needs validation.
