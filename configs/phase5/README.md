# Phase 5: Baselines, Ablations, And Reporting

Purpose:
- Run statistical baselines.
- Export score/component ablations from a completed model test output.
- Prepare final comparison artifacts.

Runnable configs:
- `statistical_baselines.yaml`
- `statistical_baselines_windows.yaml`
- `score_ablation.yaml`
- `student_t_logreturn_volume_prior_w60_ep1.yaml`
- `student_t_logreturn_volume_prior_test_w60.yaml`
- `student_t_logreturn_volume_context_attention.yaml`
- `student_t_logreturn_volume_extreme_attention_npz.yaml`

Current limitation:
- Neural and sklearn baselines from older code are not active after cleanup.
- Active baseline support is the statistical baseline suite embedded in `src/gbm/statistics.py`.
- Phase 5 attention artifacts are diagnostic only. Current evidence supports
  Student-t NLL as the detector signal, not attention as a detector.
