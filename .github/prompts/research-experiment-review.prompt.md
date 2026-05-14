---
description: "Review a research experiment design for statistical rigor, leakage risk, baseline fairness, ablation completeness, and publication readiness."
name: "Research Experiment Review Prompt"
---

Review the following research experiment design with a strict publication-quality standard.

Focus on whether the plan is scientifically defensible, statistically valid, and complete enough to support the claims in a paper.

Use [README.md](../../README.md), [AGENTS.md](../../AGENTS.md), and [docs/legacy/README.md](../../docs/legacy/README.md) as the source of truth for the current repo scope, run path, and curated research context.

Check these areas:
- Research questions and whether each one maps to a concrete experiment.
- Time-based data splitting and any leakage across train, validation, test, or rolling windows.
- Baseline coverage, fairness, and equal tuning budget.
- Metric choice for detection, early warning, and interpretability.
- Statistical testing, confidence intervals, variance reporting, and effect size.
- Ablation completeness and whether each contribution is isolated cleanly.
- Robustness across seeds, regimes, and stock subsets.
- Faithfulness or stability tests for any explanation claims.
- Whether run instructions are cross-platform and avoid hard-coding CUDA when Mac MPS support is required.

Use this output format:
- Verdict: pass / pass with notes / fail
- Critical issues, if any
- Minor issues, if any
- Missing experiments, if any
- Statistical concerns, if any
- Recommended next action

If evidence is missing, state that directly instead of inferring that the method is good.

Input to review:
{{input}}
