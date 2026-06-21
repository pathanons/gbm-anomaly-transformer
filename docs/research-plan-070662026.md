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

1. Confirm the current control points in `src/gbm/score.py`, `src/gbm/train.py`, and `src/gbm/test.py` so training loss, diagnostics, distribution-shift score, and final anomaly score are not conflated.
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

### Phase 1A: Loss Ablation Matrix

Train and compare the following loss configurations under the same data split, seed policy, and validation-only selection rules:

1. `reconstruction` only
2. `nll` only
3. `moment discrepancy` only
4. `association discrepancy` only
5. `nll + moment discrepancy + association discrepancy` with reconstruction removed
6. `nll + association discrepancy`
7. `nll + moment discrepancy`
8. `moment discrepancy + association discrepancy`

For this ablation, keep training and test-time scoring aligned with the same active terms, and report whether removing reconstruction reduces noise, sharpens the main predictive signal, and improves downstream separation without using test data for model selection.

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
38. Use YAML configs through `run.py` as the primary execution path for statistical baselines and score ablations.

## Reporting Rules

- Always report mean, standard deviation, and seed-level variation when multiple runs are available.
- Do not use test data to select thresholds, weights, or model variants.
- Keep the active GBM joint pipeline as the reference execution path.
- Update this file after each implementation batch with a short note of what changed, what failed, and what still needs validation.

## Implementation Log

### 2026-06-22 - Phase 5 Findings Consolidation

- Added `docs/phase5-findings-summary-20260622.md` as the report-facing Phase 5 finding index. Carry forward Student-t next-day NLL as the useful detector signal for endpoint log-return +/-3 std weak labels: Q98 gives 86.02% recall, Q98.5 gives 64.78% precision and 77.62% recall, and Q99 gives 78.76% precision and 65.52% recall under validation-quantile thresholding.
- Recorded the negative attention result: current attention motifs did not show a meaningful repeated return-shock pattern. Rank-1 endpoint attention never matched the max absolute-return day in the selected windows, and any top-5 attention key overlapped a top-5 absolute-return day only 15.48% of the time. Keep attention images/video/NPZ exports for future interpretability work, but do not claim attention is a detector or causal explanation from this run.
- Cleaned the Phase 5 README so the active reusable configs are explicit and old baseline limitations remain clear.

### 2026-06-17 - Fast Next-Day Student-t NLL Baseline

- Added `target: next_day_log_return` support while preserving the old window-return default. Runtime datasets now emit `target_return` only for next-day configs, and train/test NLL uses that scalar target for `p(r_{t+1}|X_t)`. Added causal volatility-aware `log_return_tail_vol` features including shifted rolling volatility and `return_z_20`; scaler fitting remains train-split only through the existing loader path.
- Added short configs `configs/phase3/student_t_nextday_nll_w60_fast.yaml` and `configs/phase3/gaussian_nextday_nll_w60_fast.yaml`: 60-day context, NLL-only score/loss, association off, no anomaly-label filtering during train, 3 epochs for a same-day evidence run. Tests passed in the training env (`19 passed`).
- Ran both configs on CUDA with `AT_OUTPUT_ROOT=D:/AnomalyTransformerRuns`. Student-t: validation mean score -2.772241, test ROC-AUC 0.995475 and PR-AUC 0.802297 against weak endpoint log-return 3-std labels. Gaussian: validation mean score -2.707452, test ROC-AUC 0.994100 and PR-AUC 0.759150. Wrote comparison artifacts to `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare`. Treat this as fast weak-label evidence only, not final proof; association, reconstruction, rolling z-score, event-window metrics, and multi-seed statistics remain open.

### 2026-06-18 - Next-Day NLL Report Figures

- Generated the fast next-day NLL visualization/evidence package via `configs/phase4/nextday_nll_report_figures.yaml` and `pipeline: nextday_report_figures`, reusing the completed Student-t and Gaussian 3-epoch runs without additional training. Outputs under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare` now include top anomalies, weak endpoint-event tables, event detection metrics, model-comparison metrics, Figure 1/2/4/5/6 ticker panels for the six most event-dense tickers, Figure 3 global NLL-vs-absolute-return scatter, Figure 7 model-comparison bar chart, and `report.md`. Student-t remains ahead of Gaussian and the rolling z-score proxy on this weak-label fast package, but reconstruction, association heatmaps for the final NLL-only model, GARCH-t, and multi-seed statistical support remain open.
- Added 109 per-ticker comparison plots under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/return_std_and_nll_by_ticker`, with endpoint log-return plus split-specific mean and +/-3 std thresholds on top and Student-t/Gaussian NLL scores below. These are explanatory plots from existing score CSVs only; no retraining or new model-selection claim was added.
- Added fixed-threshold Student-t NLL visualizations for `score > 0.03` under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/fixed_threshold_0p03_studentt_nll`. The export includes 109 per-ticker plots, `detected_anomalies.csv`, and `summary_by_ticker.csv`; this is an illustrative fixed-cutoff diagnostic, not a validation-selected threshold.
- Added circled overlap plots under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_fixed_threshold_0p03`, marking days where `score > 0.03` and endpoint log-return is outside split +/-3 std. The fixed cutoff produced 3,027 score anomalies, 1,488 weak endpoint anomalies, and 1,355 overlaps across 109 tickers.
- Added the same circled overlap diagnostic for `score > 0.01` under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_fixed_threshold_0p01`. The lower fixed cutoff produced 3,074 score anomalies and 1,360 overlaps against the same 1,488 weak endpoint anomalies.
- Added circled overlap diagnostics for fixed thresholds `score > 0.05` and `score > 0.005`. The `0.05` cutoff produced 2,990 score anomalies and 1,351 overlaps; the `0.005` cutoff produced 3,085 score anomalies and 1,361 overlaps. These remain visual threshold diagnostics, not validation-selected operating points.
- Recomputed the overlap plots using validation-quantile thresholds where `delta` is the upper-tail fraction: `0.05 -> Q95`, `0.03 -> Q97`, `0.01 -> Q99`, and `0.005 -> Q99.5`. Outputs live under `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_*`, with combined summary `validation_quantile_threshold_summary.csv`. Test-set overlaps/recalls were: delta 0.05 = 1,472/0.989 recall, delta 0.03 = 1,399/0.940, delta 0.01 = 975/0.655, and delta 0.005 = 622/0.418 against 1,488 weak endpoint anomalies.
- Added validation-quantile `delta=0.02` (`Q98`) overlap plots under the same output family and refreshed `validation_quantile_threshold_summary.csv`. Its validation threshold was 0.334968, producing 2,401 score anomalies, 1,280 overlaps, 0.533 precision, and 0.860 recall against the weak endpoint labels.
- Added validation-quantile `delta=0.015` (`Q98.5`) overlap plots and refreshed both `validation_quantile_threshold_summary.csv` and `validation_quantile_confusion_counts.csv`. Its validation threshold was 0.734406, producing 1,783 score anomalies, 1,155 overlaps, 628 false alarms, 333 missed weak endpoint anomalies, 0.648 precision, and 0.776 recall.
- Finding checkpoint: the fast Student-t next-day NLL score is useful as an anomaly-ranking signal. Validation-quantile thresholds create a clear precision/recall tradeoff against weak endpoint +/-3 std labels, with `delta=0.02` and `delta=0.015` looking like practical operating-region candidates for follow-up. Treat this as actionable diagnostic evidence for thresholding/reporting, not final baseline-superiority proof.
- Wrote the finding as `docs/research-conclusion1.md`, summarizing the Student-t next-day NLL validation-quantile threshold evidence, candidate operating points, limitations, and next actions.
- Wrote `docs/research-conclusion2.md` to fix the interpretation framing: keep Student-t NLL as the primary anomaly score and use Transformer attention maps as an explanation/diagnostic layer for selected anomaly, false-alarm, missed, and normal-control windows rather than claiming attention-based scoring beats NLL.

### 2026-06-17 - Ponytail Cleanup

- Removed orphaned legacy `scripts/gbm/*_joint.py` entry points plus duplicate `src/gbm/losses.py` and `src/gbm/scoring.py`; active loss/scoring now remains centralized in `src/gbm/score.py` through the single `run.py` / `main.py` path. Added a narrow `.gitignore` rule for generated `datasets/SP500_logreturn_volume_w*/` folders so regenerated window artifacts stay out of git. Restored robust-z visualization columns alongside raw-score columns. Compile checks passed, and `python -m pytest unittest` passed with 6 tests and 3 torch-dependent skips.
- Cleaned local generated clutter by removing ignored cache/output folders (`.conda`, `.vscode`, `results`, generated `SP500_logreturn_volume_w*`, empty `tests`, `tmp`, and stale empty `scripts/gbm`). Moved root-level experiment YAMLs into the existing phase layout: runnable best config to `configs/general`, train/model configs to `configs/phase3`, and attention/MAD diagnostics to `configs/phase4`. Dry-run checks passed for `configs/general/best.yaml` and `configs/phase4/attention_best.yaml`; compile checks passed.
- Added `association_mode=student_t_log_return` with a heavy-tailed Student-t timestamp posterior using stepwise accumulated drift/variance, plus `features=log_return_tail_vol` for return magnitude, squared return, rolling volatility, and volume context. Ran `configs/phase3/student_t_tail_vol_w60.yaml` for 5 epochs on CUDA under `D:/AnomalyTransformerRuns/experiments/student_t_tail_vol_w60_h4_l3_d128_lr0p0001_ep5_s42`; endpoint-label test ROC-AUC was 0.657241 and PR-AUC was 0.023271 across 111,993 windows with anomaly rate 0.013287. The validation p95 threshold gave F1 0.041962, sensitivity 0.047715, specificity 0.983485, and precision 0.037447. Generated ROC/PR plots and a metrics report; treat this as diagnostic evidence only, not a baseline-superiority claim.

### 2026-06-20 - Student-t NLL Attention Diagnostics Plot Fix

- Fixed attention diagnostic plotting so the log-return panel falls back to `LogReturn` or computes log return from `Close` when lowercase `log_return` is absent. Regenerated all 144 layer/head plots for the selected delta=0.015 Student-t NLL anomaly windows under `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_w60_fast_ep3_s42/reports/attention_nll_top_anomalies_delta_0p015/figures_with_log_return`; the panels now show log return with +/-3 std lines. Because this diagnostic uses `association_mode: none`, interpret the learned series attention and endpoint row; the prior/difference panels are not evidence of a GBM-prior contrast in this run.
- Regenerated a reduced attention diagnostic set with only `layer=0, head=0` under `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_w60_fast_ep3_s42/reports/attention_nll_top_anomalies_delta_0p015/figures_layer0_head0_only`. The companion `layer0_head0_attention_reading.csv` records the endpoint top key date, top lag, top weight, and a simple diffuse/moderate/spiky label for the 12 selected NLL anomaly windows.
- Added local 60-point Q99 NLL attention plots under `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_w60_fast_ep3_s42/reports/attention_nll_top_anomalies_delta_0p015/q99_local_window_attention_layer0_head0`. Each plot aligns Close, log return, next-day Student-t NLL score with the validation Q99 threshold (`1.2435514020919796`), and the learned 60x60 attention map for `layer=0, head=0`; the selected target score is placed at the input-window endpoint so the red guide line matches the endpoint query row.
- Corrected the local Q99 attention plots to use the full model export `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_w60_fast_ep3_s42/reports/test_scores.csv` for the NLL score curve, matching the source used by `overlap_circled_validation_quantile_delta_0p01`. The corrected plots live under `.../q99_local_window_attention_layer0_head0_full_test_nll`; `full_test_nll_q99_attention_summary.csv` confirms every selected score from `test_scores.csv` matches the attention manifest score.
- Reviewed the active anomaly-detection pipeline for data preparation, Transformer attention, Student-t NLL scoring, and threshold semantics. The key audit result is that `delta=0.01 = Q99` means upper-tail fraction mapped to the validation-score 0.99 quantile (`1.2435514020919796` in the current Student-t next-day NLL artifact), while literal `score > 0.01` is a separate fixed-threshold diagnostic and should not be described as Q99.
- Added and ran `configs/phase3/student_t_nextday_nll_student_prior_w60_fast.yaml`, which keeps the final score as next-day Student-t NLL (`score_mode: nll_only`) but trains the Transformer with `association_mode: student_t_log_return` and `association_weight: 0.1` so attention has a Student-t temporal prior. The run completed via checkpoint plus separate validate/test resume configs under `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_student_prior_w60_ep3_s42`; test ROC-AUC was 0.995709 and PR-AUC was 0.807734, with mean association discrepancy 0.207191.
- Exported all-layer/all-head attention diagnostics for the top 12 test-score windows via `configs/phase4/student_t_prior_attention_top_anomalies_w60.yaml`. The output contains 144 plots plus manifest under `D:/AnomalyTransformerRuns/experiments/phase4_student_t_nextday_nll_student_prior_w60_ep3_s42/reports/attention_student_t_prior_top_anomalies`, showing learned series attention, Student-t prior attention, and their absolute difference with diagonal masking.
- Added a Phase 5 diagnostic run `configs/phase5/student_t_logreturn_volume_prior_w60_ep1.yaml` using only `log_return_z` and `volume_z` features with Student-t predictive NLL plus `student_t_log_return` association prior. The one-epoch checkpoint was resumed through `configs/phase5/student_t_logreturn_volume_prior_test_w60.yaml`; test ROC-AUC was 0.995921 and PR-AUC was 0.813121 against weak endpoint +/-3 std labels, with mean association discrepancy 0.335790.
- Added `attention_select_by: extreme_normal_context` for Phase 5 diagnostics. It selects the global maximum absolute endpoint log-return reference and one normal reference closest to mean endpoint log-return, then exports each reference day plus the 10 preceding windows. `configs/phase5/student_t_logreturn_volume_context_attention.yaml` generated 22 layer-0/head-0 plots and `.npz` artifacts under `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_normal_context_lag10_layer0_head0`; each artifact now includes learned/prior/diff attention, raw/scaled features, returns, date deltas, full latent sequence, endpoint latent, and mean latent.

### 2026-06-21 - Extreme Log-Return Attention Motif Probe

- Added `attention_select_by: extreme_log_return_context` and config `configs/phase5/student_t_logreturn_volume_extreme_attention_npz.yaml` to select top endpoint log-return events whose per-ticker endpoint log-return exceeds +/-3 std, then export lag-6 through lag-0 context windows as `.npz` without retraining. The Phase 5 Student-t prior run exported 84 context windows for 12 extreme references under `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz`.
- Initial motif summary: rank-1 endpoint attention keys had median lag 8 and mean lag 8.19 from the window end. The rank-1 key matched the max absolute-return day inside the input window 0/84 times, and any top-5 attention key landed in the window's top-5 absolute-return days for 15.5% of windows. This suggests a possible temporal/context motif rather than direct attention to the largest input return; treat as hypothesis-generating only until compared across layers/heads and normal controls.
- Added readable motif reports and figures under `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz`, including `attention_motif_report_th.md`, selected-event bars, rank-1 key-lag histogram, key-lag heatmap, attention/return overlap rates, mean endpoint attention row, and mean attention matrix.

### 2026-06-16 - GBM Log-Return Likelihood Prior

- Added a GBM log-return likelihood prior mode (`association_mode: gbm_log_return_likelihood`) that builds a causal prior association from cumulative log returns inside each window using `log(S_i/S_j) ~ Normal(drift_i * elapsed_ij, sigma_i^2 * elapsed_ij)`, with learnable per-head/time drift and volatility parameters. Added `features: log_return_only` and `configs/phase3/gbm_logreturn_prior_w60_logreturn_minimax.yaml` to train on `window_size=60` log-return-only inputs with `nll_assoc_minimax` and `softmax_product` scoring. The run completed under `D:/AnomalyTransformerRuns/experiments/gbm_logreturn_prior_w60_logret_h4_l3_d128_lr0p0001_ep20_s42`; early stopping stopped at epoch 6/20. Test export contains 111,883 windows across 111 tickers with window-level anomaly rate 0.985217, ROC-AUC 0.536004, PR-AUC 0.985883, mean NLL -2.613273, and mean association discrepancy 0.924094. Treat this as an implementation/diagnostic run because current window labels are extremely dense and not suitable for final model-quality claims.
- Added phase-4 visualization configs for the GBM log-return prior run. Generated 109 all-ticker test diagnostic panels under `reports/panels_logreturn_anomaly`, highlighting daily `log_return_anomaly` regions and plotting the model score. Generated 109 layer-0/head-0 no-diagonal attention diagnostics under `reports/attention_endpoint_logreturn_anomaly_top_per_ticker/figures`, selecting one endpoint `log_return_anomaly` window per ticker from the score CSV; the attention manifest includes `true_log_return_anomaly=1` for selected examples.

### 2026-06-15 - Dataset Close-Difference Validation Histograms

- Marked `best_nllassoc_w100_h2_l4_lr0p00005_ep40_s99` as `best_model_config_final` for follow-up diagnostics. Reused its checkpoint without modifying the original experiment folder, and wrote fresh outputs to `D:/AnomalyTransformerRuns/experiments/best_model_config_final`. The retest uses the original `datasets/SP500_event_taxonomy_w100` data and labels only endpoint days whose test-split daily log return exceeds mean +/- 3 std. The new `test_scores.csv` contains 102,855 windows, 107 tickers with test scores, and 1,383 endpoint log-return anomaly positives. Added `nll_robust_z * softmax(-association_robust_z)` as `nll_robust_softmax_assoc_score`; its endpoint-label ROC-AUC was 0.657773 and PR-AUC was 0.019602. Generated 107 four-panel figures under `figures/endpoint_logreturn_robust_score_panels` with Close, log_return, original score, robust-product score, and transparent green vertical lines at endpoint log-return anomalies.
- Added a YAML-driven dataset validation visualization path (`pipeline: dataset_difference_histograms`) in `src/gbm/visualize.py` and `main.py`, with config `configs/phase1/dataset_close_difference_histograms_w100.yaml`.
- Generated `window_size=100` close-to-previous-close percent-difference histograms for all 111 discovered tickers, split into train/test endpoint dates under `D:/AnomalyTransformerRuns/findings/datasets_validation/histogram_difference_all` and anomaly-labeled endpoint dates under `D:/AnomalyTransformerRuns/findings/datasets_validation/histogram_difference_anomaly`.
- Wrote 222 PNGs per output folder plus `close_difference_histogram_summary.csv`. This is a dataset-label diagnostic only and does not tune thresholds or support model-improvement claims.
- Added an OHLCV-only feature preparation path (`pipeline: stock_feature_prepare`) that creates per-ticker `log_return`, causal `volume_z`, `rolling_volatility_20`, per-ticker normalized feature columns, rule-based labels (`jump`, `drop`, `volume_spike`, `volatility_shock`, `is_anomaly`), and compressed sliding-window arrays for `window_size` 30, 60, 90, and 128 under `datasets/SP500_logreturn_volume_w*`.
- Added `pipeline: dataset_logreturn_histograms` and generated all-ticker log-return distribution plots for all days and anomaly-labeled days under `D:/AnomalyTransformerRuns/findings/datasets_validation/histogram_logreturn_all/w*` and `D:/AnomalyTransformerRuns/findings/datasets_validation/histogram_logreturn_anomaly/w*`. These plots support dataset inspection only; the rule labels remain diagnostic until validated against external events or a formal labeling protocol.
- Revised the OHLCV-only diagnostic labels so `log_return_anomaly` is defined only from raw per-ticker `log_return` outside `mean(log_return) +/- 3*std(log_return)`, and `volume_anomaly` remains a separate label from `abs(causal volume_z) >= 3`. Regenerated `datasets/SP500_logreturn_volume_w*`; sliding-window inputs use only `log_return_z` and `volume_z`. Regenerated `histogram_logreturn_anomaly/w*` using only `log_return_anomaly == 1`, so the anomaly histograms show the log-return values on return-anomaly days rather than an OR over volume anomalies.
- Added a third diagnostic label, `high_swing`, for days where `abs(log_return) > 0.1`. Regenerated `datasets/SP500_logreturn_volume_w30`, `w60`, `w90`, and `w128` only; no histogram artifacts were regenerated for this label update.
- Corrected `log_return_anomaly` again to be split-specific: each ticker's train, validation, and test daily rows now use their own `mean(split log_return) +/- 3*std(split log_return)` thresholds. Regenerated all `datasets/SP500_logreturn_volume_w*`, added per-folder `split_log_return_thresholds.csv`, and wrote a consolidated audit to `D:/AnomalyTransformerRuns/findings/datasets_validation/label_audit/log_return_anomaly_split_audit.csv`; the audit found 0 mismatches across 1,324 ticker/split/window checks.
- Ran hotkey-01 3-epoch MLE-prior raw-NLL minimax training on `datasets/SP500_logreturn_volume_w60` with features `log_return_z` and `volume_z`, `association_mode=mle_gbm`, loss `NLL - 3*association_discrepancy`, and score `NLL * softmax(-association_discrepancy)` under `D:/AnomalyTransformerRuns/experiments/01_mle_nll_minimax_logvol_w60_h4_l3_d128_lr0p0001_ep3_s42`. Test ROC-AUC from exported window scores was 0.490034. Interpretation is limited because window-level positives are extremely dense (110,229 positives vs 1,654 negatives); daily endpoint/event evaluation is still needed before making model-quality claims. Generated ROC and A ticker test diagnostic panel under the run's `reports/roc_panel`.
- Added YAML-driven all-ticker model diagnostic panels (`configs/phase4/01_model_panels_all_tickers.yaml`) and generated 109 test-split figures under `D:/AnomalyTransformerRuns/experiments/01_mle_nll_minimax_logvol_w60_h4_l3_d128_lr0p0001_ep3_s42/reports/panels_all_tickers`. Each figure shows Close, raw log_return with split-specific +/-3 std lines, model anomaly score, and transparent green highlights on days where `log_return_anomaly` or `volume_anomaly` is true. Added all-ticker labeled attention diagnostics (`configs/phase4/01_attention_labeled_top_per_ticker.yaml`) and exported 109 selected test anomaly windows plus layer-0/head-0 no-diagonal attention plots under `reports/attention_labeled_top_per_ticker`; each ticker uses its highest-score labeled test window, so this is a qualitative attention inspection rather than a model-selection result.
- Ran a 20-epoch-request canonical Gaussian/log-return association variant on the new `log_return_z + volume_z` features (`configs/phase3/01_canonical_gaussian_minimax_logreturn_volume_20epoch.yaml`, `association_mode=gaussian_log_return`, `loss_combine_mode=nll_assoc_minimax`, `score_mode=softmax_product`). Early stopping stopped at epoch 7; test ROC-AUC from the dense window labels was 0.514249 and PR-AUC was 0.984721 under `D:/AnomalyTransformerRuns/experiments/01_canonical_gaussian_minimax_logvol_w60_h4_l3_d128_lr0p0001_ep20_s42`. Generated 109 all-ticker test panels highlighting only `log_return_anomaly` under `reports/panels_logreturn_anomaly`, plus 109 layer-0/head-0 no-diagonal attention diagnostics under `reports/attention_logreturn_anomaly_top_per_ticker`. The attention manifest confirms all 109 selected windows have `true_log_return_anomaly=1`; volume anomalies may overlap but were not the selection rule.
- Corrected the attention selection to use endpoint daily labels rather than window-level OR labels: `attention_select_by=endpoint_log_return_anomaly` now merges `{ticker}_anomaly_label.csv` by ticker and score `end_date`, so the selected attention window ends exactly on a day where raw daily `log_return` exceeds the split-specific +/-3 std threshold. Regenerated 109 attention diagnostics under `reports/attention_endpoint_logreturn_anomaly_top_per_ticker`; `endpoint_logreturn_anomaly_audit.csv` reports 109/109 endpoint `log_return_anomaly=1` and 0 mismatches.

### 2026-06-13 - Best Suite Config Selection

- Aggregated `D:/AnomalyTransformerRuns/experiments/best_nllassoc*` validation score exports into `results/best_config_seed_metrics.csv` and `results/best_config_summary.csv`.
- Selected `h=2, e_layers=4, lr=5e-5, epochs=20` as the current validation-first best config among completed runs: mean validation ROC-AUC 0.536451 and PR-AUC 0.974975 across five seeds. The matching 40-epoch config produced identical metrics, likely due to early stopping/checkpoint reuse, so the 20-epoch setting is preferred for lower compute.
- Treat test metrics as confirmation only, not model-selection evidence; validation performance differences remain small and should be reported with seed variation.
- Added `configs/general/best.yaml` as the runnable single-config version of the selected seed-42 checkpoint family (`best_nllassoc_w100_h2_l4_lr0p00005_ep20_s42`) and dry-run validated it through `run.py`.
- Added YAML-driven attention diagnostics through `configs/phase4/attention_best.yaml` and `pipeline: attention_visualize`. The path exports selected attention matrices to compressed `.npz` artifacts and renders learned-series, GBM prior, absolute-difference, and endpoint-profile plots. Generated top-5 association-discrepancy diagnostics for the selected best checkpoint under `D:/AnomalyTransformerRuns/experiments/best_nllassoc_w100_h2_l4_lr0p00005_ep20_s42/reports/attention`.
- Added `configs/phase4/attention_delta_best.yaml` to select attention windows by `abs(diff(robust_z(nll + association_discrepancy)))` per ticker. Generated top-5 delta-score spike diagnostics under `D:/AnomalyTransformerRuns/experiments/best_nllassoc_w100_h2_l4_lr0p00005_ep20_s42/reports/attention_delta`; the selected windows are anomaly-labeled and include score-z jump context in the manifest and plots.
- Verified the selected best checkpoint already has all-ticker test outputs: `test_scores.csv` contains 102,855 test windows across 107 tickers. The delta-attention diagnostic selection was run across that all-ticker score file, with top windows from ALGN, CLX, DLTR, INTC, and LW.
- Added `configs/phase4/attention_delta_A_AKAM.yaml` for overlap-only attention diagnostics on tickers A and AKAM. It keeps all-ticker loader IDs intact, filters selected windows to `delta_score_robust_z > 10*MAD` and price-anomaly starts, and generated 19 overlap plots under `reports/attention_delta_mad10_overlap_A_AKAM`.
- Extended attention plotting to support `attention_layer: all` and `attention_head: all`. Regenerated A/AKAM overlap diagnostics for all 4 layers and 2 heads, producing 152 plots under `reports/attention_delta_mad10_overlap_A_AKAM_all_layers_heads`; per-plot endpoint summaries now reflect the specific layer/head rather than the averaged manifest values.
- Added optional diagonal masking for attention diagnostics so self-attention does not dominate the heatmap color scale or endpoint profile. Regenerated the A/AKAM overlap diagnostics with `attention_mask_diagonal: true`, producing 152 no-diagonal plots under `reports/attention_delta_mad10_overlap_A_AKAM_all_layers_heads_no_diag`.
- Added `configs/phase4/attention_normal_A_AKAM.yaml` as a low-delta comparison set for A and AKAM. It selects two non-price-anomaly-start, non-MAD-spike windows per ticker with near-zero `delta_score_robust_z`, masks diagonal attention, and renders all 4 layers x 2 heads under `reports/attention_normal_low_delta_A_AKAM_no_diag`.
- Added `configs/phase4/attention_previous_A_AKAM.yaml` to inspect the x-1 window immediately before each A/AKAM k=10 MAD overlap spike x. The manifest records the selected previous window plus the reference spike window/date/delta-z, and renders all 4 layers x 2 heads with diagonal masking under `reports/attention_previous_delta_mad10_overlap_A_AKAM_all_layers_heads_no_diag`.
- Added `attention_score_formula: product` support for delta-attention diagnostics and `configs/phase4/attention_product_A_AKAM.yaml`. The product diagnostic uses `nll * association_discrepancy` before robust-z/delta/MAD filtering; for A/AKAM overlap filtering it selected 7 windows and produced 56 all-layer/head no-diagonal plots under `reports/attention_product_mad10_overlap_A_AKAM_all_layers_heads_no_diag`. Interpret with care because exported `nll` values are negative in the current score files.
- Added `loss_combine_mode: nll_assoc_product` and `configs/phase3/product_nllassoc_3epoch.yaml`, then trained a 3-epoch product-loss experiment (`product_nllassoc_w100_h2_l4_lr0p00005_ep3_s42`). The run completed train/validate/test; test ROC-AUC was 0.599841 and PR-AUC 0.992245. Product training drove mean association discrepancy high (`17.56`), consistent with the risk that multiplying negative NLL by positive association can incentivize larger association discrepancy rather than regularizing it.
- Generated product-score MAD and attention diagnostics for the 3-epoch product-loss model on A and AKAM. The original sum-score diagnostic produced no k=10 MAD spikes for these two tickers, so these plots use the product score consistently with the product-loss experiment; this yielded A=2 model spikes and AKAM=11 model spikes, with no-diagonal all-layer/head attention plots for spike windows, previous x-1 windows, and low-delta normal comparison windows.
- Added an element-wise `nll_assoc_softmax_product` training/scoring path matching `Softmax(-AssDis) * NLL` at the timestamp level, plus `score_mode: softmax_product` and `score_formula: softmax_product` for test export and MAD visualization. Ran `configs/phase3/product_softmax_nllassoc_20epoch_k3_all.yaml` end-to-end for all tickers: train/validate/test completed, test ROC-AUC was 0.582151 and PR-AUC 0.991541, and k=3 MAD visualization produced 107 ticker plots with 11,728 model spikes, 2,734 price anomaly starts, and 972 overlaps. Treat as a diagnostic run; k=3 is intentionally sensitive and not validation-selected.
- Added `association_mode: mle_gbm`, a deterministic GBM prior built from the per-window irregular-time MLE volatility estimator shown in the GBM reference formula. The learned series attention is compared to this MLE transition prior with the existing symmetric KL association discrepancy. Added `loss_combine_mode: nll_assoc_minimax` for a two-phase training objective using `|NLL| + lambda|AssDis|` to pull learned attention toward the MLE prior, then `|NLL| - lambda|AssDis|` with detached prior for the maximizing phase. Added `configs/phase3/mle_prior_nllassoc_20epoch_k3_all.yaml` for an all-ticker 20-epoch train/validate/test run with final `Softmax(-AssDis) * NLL` scoring and k=3 MAD visualization; dry-run passes, but the full MLE-prior minimax training run has not yet been executed.
- Revised the MLE experiment to match the clarified objective: the transformer predicts GBM log-drift and volatility parameters, those parameters are supervised against per-window MLE targets from the irregular-time volatility formula, and the learned prior association is compared to series attention with symmetric KL. Added `loss_combine_mode: mle_param_minimax`, `score_mode: mle_param_softmax_product`, and `configs/phase3/mle_param_minimax_3epoch_k3_all.yaml` (`window_size=100`, `d_model=512`, `heads=8`, `layers=3`, `lambda=3`, `epochs=3`). The full train/validate/test/visualize run completed under `D:/AnomalyTransformerRuns/experiments/mle_param_minimax_w100_h8_l3_d512_lr0p0001_ep3_s42`; test ROC-AUC was 0.460064 and PR-AUC 0.988875, with k=3 MAD plots for 107 tickers, 6,663 model spikes, 2,734 price anomaly starts, and 848 overlaps. Treat as a diagnostic implementation run, not evidence of model improvement.
- Updated MAD visualization to remove robust-z panels and display the raw final score as its own subplot, separate from the nll/association component subplot. Added `configs/phase4/mle_param_minimax_mad_raw_k3_all.yaml` and regenerated raw-score k=3 plots under `figures/mad_k3_raw_score` for the completed MLE-parameter minimax run.
- Added `features: close_only` and raw-level MAD thresholding (`threshold_mode: raw`) so diagnostic plots can threshold directly on the raw final score rather than delta score. Ran `configs/phase3/mle_param_minimax_close_only_5epoch_raw_k3_all.yaml` (`Close` input only, 5 epochs, k=3 raw-score MAD). The run completed train/validate/test/visualize under `D:/AnomalyTransformerRuns/experiments/mle_param_minimax_close_w100_h8_l3_d512_lr0p0001_ep5_s42`; test ROC-AUC was 0.469353 and PR-AUC 0.989040, with 107 ticker plots, 5,374 model spikes, 2,734 price anomaly starts, and 170 overlaps. Treat as diagnostic evidence only.

### 2026-06-10 - Phase Config Reorganization

- Reorganized runnable YAMLs into `configs/phase1` through `configs/phase5` so data preparation, statistics, insights, model/loss/score ablations, visualization, and baseline experiments are phase-addressable through `python run.py --config ...`.
- Added `configs/general` for routine data preparation, train, test, and k=9 MAD visualization entry points independent of phase history.
- Kept execution centralized in `run.py` / `main.py`; deprecated split runner paths now point to phase configs.
- Dry-run checked all current YAML configs after the reorganization. All configs parsed successfully; `association_none.yaml` now quotes `"none"` so it is preserved as a literal model option.
- Updated `AGENTS.md`, `README.md`, `.codex/skills/gbm-anomaly-transformer/SKILL.md`, `.github` agent/instruction files, and `docs/AGENT_CONTEXT_AUDIT.md` to point agents at the lean YAML-driven workflow instead of old `scripts/gbm` runners. Synced the repo-local Codex skill to the global user skill.
- Added `docs/research-note-20260610.md` to capture the current research-status discussion: data, model, loss, training, scoring, baselines, ablations, robust z-score, k=9 MAD, and unsupported claims that still need evaluation.
- Added `docs/reports/weekly-loss-threshold-report-20260610.txt` as a Thai plain-text report summary explaining loss/score component diagnostics, scale incompatibility, robust z-score, delta-score MAD thresholding, and why k=9 is diagnostic rather than a final selected threshold. No new experimental claim was introduced.
- Expanded `docs/reports/weekly-loss-threshold-report-20260610.txt` with an evidence checklist listing the raw CSV/JSON summaries, representative ticker plots, k=3 vs k=9 comparison artifacts, and missing validation/baseline/seed files needed to support or strengthen the report claims.
- Copied the week-1 report evidence package to `D:/AnomalyTransformerRuns/weekly/week1`, including repo reports, component-noise CSV/JSON summaries and representative plots, MAD k=9 summaries/plots, and available MAD k=3 comparison files. The manifest records that validation-threshold sweep, locked test predictions, baseline comparison, and multi-seed summaries are still missing future evidence files.
- Reviewed `jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9` artifacts and added `docs/reports/mad-k9-jumpdrop-diagnostic-report-20260610.md`. k=9 reduced model spikes from 12,375 at k=3 to 3,336 while raising overlap-per-spike from 19.70% to 42.18%; coverage fell from 89.17% to 51.46%. Treat as useful conservative diagnostic evidence, not final detector proof.
- Reviewed `phase2_diststudy_legacy_model_score_legacy` score-shape artifacts and added `docs/reports/phase2-loss-component-noise-report-20260610.md`. Evidence shows raw legacy score is dominated by scale-unstable reconstruction outliers in some tickers (`reconstruction_error` median 0.122771 vs max 7469.848145; total legacy score max 7467.566895), supporting the decision to remove/down-weight noisy components from the final anomaly score while keeping training-loss ablation as an open question.
- Consolidated `src/gbm` helper modules into the canonical stage files. Active code now lives in `datasets.py`, `model.py`, `score.py`, `train.py`, `test.py`, `visualize.py`, `statistics.py`, and `__init__.py`; removed separate helper files for data/features/paths/io/losses/scoring/metrics/embedding/device/runtime/training/diagnostics/plotting. Compile and general train/visualize dry-runs passed after the merge.

- 2026-06-07: Started Phase 1 implementation. Added a refactored distribution-shift helper in `src/gbm/losses.py`, added `loss_mode` to `scripts/gbm/train_joint.py` so the GBM model can be trained with legacy vs refactored discrepancy loss, and extended `scripts/gbm/validate_joint.py` / `scripts/gbm/test_joint.py` to export both `legacy_score` and `refactored_score` for shape comparison without thresholding.
- 2026-06-07: Syntax validation passed on the touched Python files with no errors reported.
- 2026-06-07: Completed all-ticker legacy-vs-refactored visualization and ticker-level noise-signal diagnostics. Current evidence favors legacy as Phase 1 default: refactored was rougher on 90/107 tickers and had weaker high-move separation on 81/107 tickers. Advisor-facing summary prepared in `docs/reports/phase1-legacy-vs-refactored-advisor-report.md`.
- 2026-06-07: Started Phase 2 implementation. Added `quantile_wasserstein_score` and `tail_weighted_quantile_wasserstein_score` in `src/gbm/losses.py`, wired these metrics into `src/gbm/scoring.py` with new score modes (`qw2`, `qw2_tail`), and extended `scripts/gbm/validate_joint.py` / `scripts/gbm/test_joint.py` with CLI controls (`quantile_count`, `tail_weight_gamma`, `tail_weight_power`) and summary outputs for the new distribution-shift candidates.
- 2026-06-07: Completed Phase 2 run comparison artifacts under `D:/AnomalyTransformerRuns/experiments/phase2_diststudy_legacy_model_score_qw2_tail/visualizations_phase2_compare` using a now-retired Phase 2 comparison helper. Global metrics are nearly identical across modes (legacy ROC-AUC 0.564463 vs QW2 0.564464 vs QW2Tail 0.564464; PR-AUC all about 0.990874). Per-ticker changes are effectively tied for 49/50 valid tickers, with one small regression on XOM for both QW2 variants.
- 2026-06-07: Decision checkpoint after Phase 1 + Phase 2. Keep `legacy` as the default score/loss baseline for subsequent work. Evidence: Phase 1 favored legacy in broad ticker-level shape diagnostics, and Phase 2 alternatives (`qw2`, `qw2_tail`) showed no statistically meaningful gain over legacy (sign-test p-values = 1.0 with only 1/50 non-zero ticker delta).
- 2026-06-07: Added the next loss-ablation experiment matrix to Phase 1A. Planned train-time comparisons now include `reconstruction`, `nll`, `moment discrepancy`, `association discrepancy`, `nll+moment+association` without reconstruction, `nll+association`, `nll+moment`, and `moment+association`, with validation-only selection and aligned training/test scoring.
- 2026-06-09: Generated k=9 MAD outlier plots for `jumpdrop_w100_thres1_nllassoc` under `D:/AnomalyTransformerRuns/experiments/jumpdrop_w100_thres1_nllassoc/thres_outlier_mad_k9`. Relative to the existing k=3 plot summary, model spikes decreased from 12,375 to 3,336 and overlaps with price anomaly starts decreased from 2,438 to 1,407 across 107 tickers. This is a visualization diagnostic only; do not treat k=9 as selected without validation-only threshold evaluation.
- 2026-06-09: Cleaned and renamed the final MAD diagnostic display path to `scripts/gbm/diagnostics/visualize_mad_spikes.py`. The script now exposes only `robust-z(nll + association_discrepancy)`, absolute first difference, and `median + 9 * 1.4826 * MAD` spike threshold. Removed legacy fixed-z/IQR/std/per-term-sum branches and the DataFrame attribute side channel.
- 2026-06-09: Removed unused one-off `run_*.bat`, `scripts/gbm/run_night_*.py`, and retired `scripts/gbm/visualize_*.py` helper scripts after consolidating the final display path around `visualize_mad_spikes.py`. Kept active runners and baseline entry points (`run_joint`, Gaussian/canonical wrappers, baseline/vanilla runners) plus active visualization scripts.
- 2026-06-09: Extracted reusable diagnostic helpers into `src/gbm/diagnostics.py` for robust z-scoring, MAD thresholds, final score/delta construction, ticker parsing, price loading, and price anomaly starts. Refactored diagnostics scripts to use these helpers, then removed obsolete one-off diagnostic plot/print scripts that duplicated the final k=9 MAD display path.
- 2026-06-09: Leaned the script layout by moving baseline helpers to `scripts/gbm/baselines/`, vanilla baseline pipeline scripts to `scripts/gbm/vanilla/`, and diagnostics CLIs to `scripts/gbm/diagnostics/`. Shared flat-YAML parsing now lives in `src/gbm/config_runner.py`, shared plotting interval helpers live in `src/gbm/plotting.py`, and fast score export is handled by `scripts/gbm/test_joint.py --fast-export` instead of a separate one-off script.
- 2026-06-09: Consolidated duplicate YAML runners into one dispatcher, `scripts/gbm/run_from_yaml.py`, controlled by `pipeline: joint` or `pipeline: vanilla` in the flat YAML config. Removed the separate joint/vanilla YAML runners, updated example configs, and verified dry-run command generation for the example, multi-window joint config, and vanilla config.
- 2026-06-09: Centralized duplicated joint runtime and command construction into `src/gbm/runtime.py`, `src/gbm/training.py`, and `src/gbm/commands.py`. `train_joint.py`, `validate_joint.py`, `test_joint.py`, and `run_joint.py` are now thinner entry points over shared loader/model/checkpoint/score/train helpers, and `loss_mode` is configurable through the unified YAML path.
- 2026-06-10: Started repository-wide collapse to a single YAML-driven entry path. Added root `main.py` and `run.py`, moved the active GBM train/validate/test execution path into canonical `src/gbm/train.py` and `src/gbm/test.py`, added YAML configs for canonical/log-return/baseline/score-ablation/MAD diagnostics, and removed old top-level run launchers plus the direct GBM `_joint` Python wrappers. Remaining migration work: move vanilla, baseline, and diagnostic script backends into `model.py`, `statistics.py`, and `visualize.py` before deleting the remaining script backends.
- 2026-06-10: Removed the remaining separate script folders for baseline, diagnostic, and vanilla execution. `main.py` now dispatches internally to canonical stage files only; MAD visualization lives in `src/gbm/visualize.py`, statistical baselines and score ablation live in `src/gbm/statistics.py`, and experiment selection is represented by YAML configs rather than runner filenames. The old vanilla-specific Python backend was removed from the active path; remaining baseline support is the statistical baseline suite now embedded in `statistics.py`.
- 2026-06-10: Added stage-specific YAML configs for `data_prepare`, `train`, `test`, and `visualize`, plus experiment configs for baseline evaluation, loss ablation, model ablation, score ablation, and MAD visualization. Verified dry-run dispatch through the single `run.py` entry point for each config family.
- 2026-06-10: Added a one-command best-model experiment suite at `configs/general/best_model_suite.yaml` and `pipeline: experiment_suite` in `main.py`. The suite runs the current best-known Gaussian log-return / NLL+association setup through train -> validate -> test, sweeps `n_heads`, encoder layers, learning rate, epochs, and seeds for stability, then generates MAD threshold visualizations for `k >= 9` (`9,10,11,12,15`). `src/gbm/visualize.py` now accepts configurable MAD `k` instead of hardcoding k=9. Added pytest coverage under `unittest/`; `python -m pytest unittest`, `py_compile`, and `python run.py --config configs/general/best_model_suite.yaml --dry-run` all pass in the appropriate environment. Treat the seed sweep as a stability/CV proxy until true walk-forward cross-validation is implemented.
- 2026-06-10: Moved the fast pre-run test gate to `unittest/` for pytest. The suite covers config parsing, YAML pipeline validation, experiment-suite dry-run expansion, MAD visualization, synthetic dataset/window construction, model/attention shape checks, score helpers, and statistical baseline summaries. Current lightweight Python environment result: `python -m pytest unittest` passes with 6 tests and skips 3 torch-dependent modules because PyTorch is not installed in that interpreter; those tests run normally in the training environment.
- 2026-06-10: Added `best.bat` as the Windows launcher for the best-model suite. Default behavior is pytest gate -> suite dry-run -> full long run with `AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns` and `DEVICE=auto`; supported checks are `best.bat --test-only`, `best.bat --dry-run`, and `best.bat --skip-tests`. Verified `cmd /c best.bat --test-only` and `cmd /c best.bat --dry-run` both pass in the current environment.
