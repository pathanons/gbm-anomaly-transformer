# Phase 1: Data And Loss Refactor

Purpose:
- Prepare event-taxonomy windows from existing datasets.
- Recheck data/statistical signals.
- Compare legacy vs refactored divergence behavior.
- Run loss-term ablations.

Runnable configs:
- `data_prepare.yaml`
- `statistical_evaluation.yaml`
- `insight_from_data.yaml`
- `legacy_score.yaml`
- `refactored_score.yaml`
- `loss_ablation_nll_assoc.yaml`

Legacy evidence:
- EXP1 dataset validation and leakage checks live under `docs/legacy/experiment-insights`.
- Phase 1 advisor report lives at `docs/reports/phase1-legacy-vs-refactored-advisor-report.md`.
