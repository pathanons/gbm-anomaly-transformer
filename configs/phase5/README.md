# Phase 5: Baselines, Ablations, And Reporting

Purpose:
- Run statistical baselines.
- Export score/component ablations from a completed model test output.
- Prepare final comparison artifacts.

Runnable configs:
- `statistical_baselines.yaml`
- `statistical_baselines_windows.yaml`
- `score_ablation.yaml`

Current limitation:
- Neural and sklearn baselines from older code are not active after cleanup.
- Active baseline support is the statistical baseline suite embedded in `src/gbm/statistics.py`.
