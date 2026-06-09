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

### 2026-06-10 - Phase Config Reorganization

- Reorganized runnable YAMLs into `configs/phase1` through `configs/phase5` so data preparation, statistics, insights, model/loss/score ablations, visualization, and baseline experiments are phase-addressable through `python run.py --config ...`.
- Added `configs/general` for routine data preparation, train, test, and k=9 MAD visualization entry points independent of phase history.
- Kept execution centralized in `run.py` / `main.py`; deprecated split runner paths now point to phase configs.
- Dry-run checked all current YAML configs after the reorganization. All configs parsed successfully; `association_none.yaml` now quotes `"none"` so it is preserved as a literal model option.
- Updated `AGENTS.md`, `README.md`, `.codex/skills/gbm-anomaly-transformer/SKILL.md`, `.github` agent/instruction files, and `docs/AGENT_CONTEXT_AUDIT.md` to point agents at the lean YAML-driven workflow instead of old `scripts/gbm` runners. Synced the repo-local Codex skill to the global user skill.
- Added `docs/research-note-20260610.md` to capture the current research-status discussion: data, model, loss, training, scoring, baselines, ablations, robust z-score, k=9 MAD, and unsupported claims that still need evaluation.
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
