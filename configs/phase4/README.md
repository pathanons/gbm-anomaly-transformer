# Phase 4: Thresholding And Visualization

Purpose:
- Reproduce MAD-based score-delta visualization diagnostics.
- Keep current k=9 MAD display config close to prior jump/drop experiments.

Runnable configs:
- `visualize_mad_k9.yaml`
- `mad_k9_jumpdrop.yaml`
- `mad_k9_from_test_scores.yaml`

Legacy evidence:
- EXP4 score-change dynamic-threshold sweeps (`w20 k=1.5`, `w50 k=3.5`) live under `docs/legacy/gbm-report-summaries`.
- The old sweep runner is not currently ported after repo cleanup.
