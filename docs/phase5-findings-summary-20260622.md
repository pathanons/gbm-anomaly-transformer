# Phase 5 Findings Summary

Date: 2026-06-22

## Finding 1: Student-t NLL is the useful detector signal

The strongest useful Phase 5 result is that Student-t negative log likelihood
detects endpoint log-return outliers well when the weak label is
`abs(endpoint log return) >= 3 std` within the split/ticker context.

This should be reported as:

> Student-t next-day NLL is a strong anomaly-ranking signal for large endpoint
> log-return moves. It gives high recall at looser validation-quantile
> thresholds and high precision at stricter validation-quantile thresholds.

Best report-ready operating points:

| validation cutoff | threshold | score anomalies | weak +/-3 std anomalies | overlaps | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q98 (`delta=0.02`) | 0.334968 | 2,401 | 1,488 | 1,280 | 53.31% | 86.02% |
| Q98.5 (`delta=0.015`) | 0.734406 | 1,783 | 1,488 | 1,155 | 64.78% | 77.62% |
| Q99 (`delta=0.01`) | 1.243551 | 1,238 | 1,488 | 975 | 78.76% | 65.52% |

Model-level comparison from the fast weak-label package:

| model | ROC-AUC | PR-AUC | top-1% precision |
| --- | ---: | ---: | ---: |
| Student-t NLL Transformer | 0.995475 | 0.802297 | 81.43% |
| Gaussian NLL Transformer | 0.994100 | 0.759150 | 77.50% |
| Rolling z-score proxy | n/a | n/a | 77.68% |

Use this as diagnostic evidence, not a final baseline-superiority claim. The
label is still a weak endpoint +/-3 std label, and the result still needs
multi-seed runs, stronger baselines, and statistical tests before paper-level
claims.

## Finding 2: Attention did not reveal a reliable return-shock pattern

The current attention diagnostics did not find a meaningful repeated pattern
that separates extreme log-return contexts from normal behavior.

Report it as:

> In this Phase 5 pass, attention matrices were useful as saved diagnostic
> artifacts, but they did not provide evidence of a statistically meaningful
> pattern explaining or detecting endpoint log-return anomalies.

Measured attention motif result:

| metric | value |
| --- | ---: |
| rank-1 attention key equals max `abs(return)` day | 0.00% |
| rank-1 attention key is in top-5 `abs(return)` days | 2.38% |
| any top-5 attention key is in top-5 `abs(return)` days | 15.48% |
| rank-1 key median lag from window end | 8 days |
| rank-1 key mean lag from window end | 8.19 days |

Interpretation:

- Do not claim attention is a detector.
- Do not claim attention is a causal explanation.
- Keep the saved attention images, videos, manifests, and NPZ files as future
  interpretability evidence.
- Future work should test matched normal controls, all layers/heads, seed
  stability, and deletion/insertion faithfulness before making explanation
  claims.

## Evidence Table

| purpose | file or folder |
| --- | --- |
| Main NLL finding writeup | `docs/research-conclusion1.md` |
| Attention framing writeup | `docs/research-conclusion2.md` |
| Student-t vs Gaussian vs rolling z-score metrics | `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/metrics_summary.csv` |
| Validation-quantile threshold table | `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/validation_quantile_threshold_summary.csv` |
| Validation-quantile confusion counts | `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/validation_quantile_confusion_counts.csv` |
| Per-threshold overlap plots | `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_*` |
| Phase 5 Student-t log-return+volume run report | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/evaluation_report.md` |
| Phase 5 scalar metrics | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/score_metrics.json` |
| Phase 5 component metrics | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/component_metrics.csv` |
| Attention motif report | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz/attention_motif_report.md` |
| Attention motif Thai report | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz/attention_motif_report_th.md` |
| Attention overlap rates | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz/attention_return_overlap_rates.csv` |
| Attention manifest and NPZ exports | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_logreturn_top12_lag6_npz` |
| Video-friendly fixed-scale attention figures | `D:/AnomalyTransformerRuns/experiments/phase5_student_t_logreturn_volume_prior_w60_ep1_s42/reports/attention_extreme_normal_context_lag10_layer0_head0/figures` |

## Configs To Keep

These are the useful future-facing Phase 5 configs:

| config | use |
| --- | --- |
| `configs/phase5/student_t_logreturn_volume_prior_w60_ep1.yaml` | short Student-t log-return+volume training run |
| `configs/phase5/student_t_logreturn_volume_prior_test_w60.yaml` | resume test/evaluation export from checkpoint |
| `configs/phase5/student_t_logreturn_volume_context_attention.yaml` | extreme-vs-normal context attention plots with fixed-scale figures |
| `configs/phase5/student_t_logreturn_volume_extreme_attention_npz.yaml` | top extreme endpoint-return NPZ/latent attention export |
| `configs/phase5/score_ablation.yaml` | reusable score/component ablation |
| `configs/phase5/statistical_baselines*.yaml` | statistical baseline support |

## Report Sentence

Student-t NLL is the Phase 5 result worth carrying forward: it detects large
endpoint log-return outliers under validation-quantile thresholding, with Q98
giving 86.02% recall and Q98.5 giving 64.78% precision / 77.62% recall against
weak +/-3 std labels. Attention diagnostics should be retained as artifacts,
but this run did not find a significant attention pattern; the current evidence
does not support using attention as a detector or causal explanation.
