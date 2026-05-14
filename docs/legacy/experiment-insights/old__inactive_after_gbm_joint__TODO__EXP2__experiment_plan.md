# EXP2 Power Prior Diagnostic Experiment Plan

## Objective
Test whether the current power-law attention prior is actually useful for the SP500 event-taxonomy task, or whether it should be replaced by a more flexible prior design.

This experiment does not try to prove that power-law is globally better. It tries to diagnose *why* the prior may fail and whether it still has a usable niche.

## Working Hypotheses
1. Power-law priors are more suitable for long-memory dynamics than for short, event-driven anomalies.
2. The current implementation may be too weak because power-law only modifies the attention prior, while the main training loss still comes from reconstruction.
3. Distance clamping may reduce contrast between near and far time steps.
4. Batch-wise normalization and validation-based thresholding may hide differences between gaussian and power-law behavior.

## Experimental Matrix

| Variant | Prior | normalize-batch | Purpose |
| --- | --- | --- | --- |
| V1 | gaussian | on | Reference baseline |
| V2 | powerlaw | on | Current target assumption |
| V3 | gaussian | off | Preprocessing control |
| V4 | powerlaw | off | Preprocessing control for power-law |

Optional slices for the same variants:
- per ticker
- per seed
- per event type
- per regime
- per window type: normal vs anomaly

## Files and Scripts

### Existing files reused by EXP2
- [src/model/attn.py](../../src/model/attn.py): defines gaussian and power-law priors, including `sigma` and `alpha` projections.
- [src/model/AnomalyTransformer.py](../../src/model/AnomalyTransformer.py): returns `series_list`, `prior_list`, and prior parameters from the encoder.
- [scripts/02_validate.py](../../scripts/02_validate.py): current reconstruction scoring path; should remain the source for anomaly scores.
- [scripts/03_test.py](../../scripts/03_test.py): current metric computation path; should be extended for EXP2 aggregation if needed.

### New scripts to add for EXP2
- [scripts/experiments/run_experiment2.py](../../scripts/experiments/run_experiment2.py): orchestrates the EXP2 variant matrix and writes the run manifest.
- [scripts/analysis/export_prior_diagnostics.py](../../scripts/analysis/export_prior_diagnostics.py): runs inference and exports `alpha`, `sigma`, attention, and prior tensors for each window.
- [scripts/analysis/compute_prior_divergence.py](../../scripts/analysis/compute_prior_divergence.py): computes KL, JS, entropy, and support-size metrics for attention/prior pairs.
- [scripts/analysis/summarize_exp2.py](../../scripts/analysis/summarize_exp2.py): aggregates ticker/seed/event/regime summaries into tables and figures.

If the analysis scripts are not added immediately, the same responsibilities can be staged inside `scripts/02_validate.py` behind an `--export-diagnostics` flag, but the diagnostic path should stay separate from the main benchmark path.

## Output Directory Layout

Use a dedicated run directory so EXP2 does not overwrite EXP1 artifacts:

`results/experiments/experiment2_power_prior_diag/`

Expected subfolders:
- `models/`: checkpoints for each variant.
- `scores/`: per-window anomaly score CSVs.
- `reports/`: JSON and Markdown summaries.
- `diagnostics/`: alpha/sigma/attention/prior exports.
- `figures/`: plots for divergence and parameter distributions.

Expected report files:
- `experiment2_runs.csv`: per ticker/seed/variant raw results.
- `experiment2_summary.csv`: mean/std summary by variant.
- `diagnostics_summary.json`: consolidated diagnostic statistics.
- `diagnostics_summary.md`: human-readable interpretation.
- `event_slice_metrics.csv`: metrics by event type.
- `normalization_sensitivity.csv`: gaussian vs power-law with and without batch normalization.

## Metrics to Collect

### 1. Prior Parameter Distribution Metrics
Collect these separately for gaussian and power-law runs:
- `sigma_mean`, `sigma_std`, `sigma_median`, `sigma_iqr`
- `alpha_mean`, `alpha_std`, `alpha_median`, `alpha_iqr`
- `prior_entropy`
- `effective_support = exp(entropy)`
- `head_variance`
- `seed_cv` for stability across seeds

Interpretation targets:
- near-uniform prior: entropy close to `log(window_size)` and high effective support
- overly local prior: low entropy and most mass concentrated in very short lags
- noisy prior: high seed variance or head variance without metric gain

### 2. Attention/Prior Divergence Metrics
Compute divergence between the learned attention distribution and the prior distribution:
- `kl_series_prior`
- `kl_prior_series`
- `js_divergence`
- `sym_kl = 0.5 * (kl_series_prior + kl_prior_series)`
- `attention_entropy`
- `prior_entropy`
- `entropy_gap = attention_entropy - prior_entropy`

These should be reported:
- per window
- aggregated by ticker
- aggregated by seed
- aggregated by event type
- aggregated by regime
- separately for normal and anomaly windows

### 3. Event-Type Slice Metrics
Report the following for each label family:
- `jump`
- `drop`
- `volume_spike`
- `volatility_shock`
- `regime_shift`

For each slice, collect:
- ROC-AUC
- PR-AUC
- F1
- precision
- recall / sensitivity
- mean lead-time proxy if available
- `alpha` / `sigma` summaries
- divergence summaries

### 4. Normalization Sensitivity Metrics
Compare `normalize-batch` on vs off:
- change in `alpha` / `sigma` distributions
- change in prior entropy
- change in divergence
- change in slice metrics by event type
- change in anomaly-score separation

### 5. Threshold Sensitivity Metrics
Do not tune on test.
Report how sensitive the prior conclusions are to threshold choice:
- current 3-sigma rule
- at least one alternative threshold derived from validation only

## How to Collect `alpha`, `sigma`, and Attention/Prior Tensors

### Required model behavior
The model already exposes these values:
- gaussian runs return `sigma` as the prior parameter.
- power-law runs return `alpha` as the prior parameter.
- `series_list` and `prior_list` are returned by the encoder stack.

### Collection procedure
1. Load the checkpoint.
2. Run the validation and/or test loader in inference mode.
3. Capture the returned `series_list`, `prior_list`, and prior parameter for each layer.
4. Reduce each tensor to per-window summaries while also preserving raw tensors for a small curated sample.
5. Write one row per window per layer to a CSV or Parquet file.

### Recommended row schema
- `ticker`
- `seed`
- `model`
- `split`
- `window_id`
- `layer_id`
- `head_id`
- `window_label`
- `regime`
- `is_anomaly`
- `event_type`
- `alpha_mean`
- `alpha_std`
- `sigma_mean`
- `sigma_std`
- `prior_entropy`
- `attention_entropy`
- `kl_series_prior`
- `kl_prior_series`
- `js_divergence`
- `effective_support_prior`
- `effective_support_attention`
- `topk_mass_prior`
- `topk_mass_attention`

If the tensor is unavailable for a specific model family, write `null` for that field rather than coercing it into a fake value.

## Analysis Logic

### 1. Normal vs anomaly windows
Compare parameter distributions and divergences between normal windows and anomaly windows.

### 2. Event-taxonomy slices
Split the analysis by the event labels already used in the SP500 taxonomy dataset.

### 3. Regime slices
Split the analysis by regime to check whether power-law behaves differently in crisis-like periods.

### 4. Normalization slices
Compare the same metrics with and without batch normalization.

### 5. Failure-mode classification
Classify the prior behavior into one of four buckets:
- uniform collapse
- overly local collapse
- noisy instability
- meaningful separation

## Statistical Reporting Standard
- Report mean and standard deviation across seeds.
- Report confidence intervals when possible.
- Use paired tests across seeds or across tickers for model comparisons.
- Do not claim improvement from a single run.
- If the result is suggestive but not significant, call it a trend.

## Decision Rules

### Keep power-law if
- it improves at least one meaningful slice in a statistically defensible way,
- the improvement survives normalization and threshold sensitivity checks,
- and it shows a non-degenerate divergence pattern rather than collapsing to gaussian-like behavior.

### Replace power-law with a more flexible prior if
- it is indistinguishable from gaussian across most slices,
- it only helps in a narrow regime-specific way,
- or its apparent gain disappears after the diagnostic checks.

## Candidate Follow-Up Modifications
If EXP2 shows that power-law is only partially useful, the next design candidates are:
- regime-conditioned prior
- piecewise local-plus-heavy-tail prior
- feature-specific prior for OHLCV vs returns
- event-type-aware prior
- mixture prior with a learned gate between gaussian and power-law

## Deliverable Sequence
1. Run the EXP2 variant matrix.
2. Export `alpha`, `sigma`, attention, and prior tensors.
3. Compute divergence and entropy metrics.
4. Slice results by event type, regime, and normalization condition.
5. Summarize whether power-law is valid, partially valid, or not worth keeping.
6. Only after that, decide whether to modify the prior family.
