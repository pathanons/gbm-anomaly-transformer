---
description: "Checklist for writing and reviewing publication-grade research experiment plans."
applyTo: "{README.md,AGENTS.md,docs/legacy/**/*.md,scripts/gbm/**/*.py,src/gbm/**/*.py}"
---

When writing or reviewing research experiment plans for this project, enforce the following standards:

- Every experiment must answer a stated research question.
- All data splits must be time-based and leakage-free.
- Thresholds, model selection, and hyperparameters must be chosen on validation data only.
- Baselines must be representative and tuned under a comparable budget.
- Results must be reported with mean, standard deviation, and confidence intervals when multiple runs are available.
- Baseline comparisons must use statistical tests; do not rely on single numbers alone.
- Ablation studies must isolate one contribution at a time.
- Early warning claims must include lead-time or delay metrics.
- Interpretability claims must include faithfulness, stability, or other direct explanation checks.
- Robustness must be tested across seeds, regimes, and stock subsets when relevant.
- Do not weaken the evaluation standard to make results look better.
- Run instructions should use the active GBM joint pipeline and `DEVICE=auto` unless a specific accelerator is required.
- For Mac M1/M2/M3, prefer the `environment.macos-mps.yml` environment and `DEVICE=mps`.

If a plan lacks statistical support, mark it as incomplete rather than acceptable.
