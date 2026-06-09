# Agent And Instruction Audit

Audit date: 2026-06-10

## Current Project State

The active repository is `gbm-anomaly-transformer`. The active implementation is no longer a collection of one-off runner scripts. The current project shape is a lean YAML-driven pipeline:

```text
run.py -> main.py -> src/gbm/<stage>.py
```

The active stage files are:

- `src/gbm/datasets.py`: data preparation and joint manifests.
- `src/gbm/model.py`: model definitions.
- `src/gbm/score.py`: loss and scoring definitions.
- `src/gbm/train.py`: training.
- `src/gbm/test.py`: validation, testing, and score export.
- `src/gbm/visualize.py`: k=9 MAD final visualization.
- `src/gbm/statistics.py`: statistical evaluation, baselines, and score ablation.

Experiment variants should be added as YAML configs, not as new Python entry-point files.

## Config Layout

- `configs/general/`: routine data preparation, train, test, and k=9 MAD visualization configs.
- `configs/phase1/`: data preparation, data statistics, insight configs, legacy/refactored score checks, and loss ablation.
- `configs/phase2/`: distribution-shift score variants (`legacy`, `refactored`, `qw2`, `qw2_tail`).
- `configs/phase3/`: model and association-mode experiments (`gaussian_log_return`, `canonical_gbm`, `temporal`, `none`).
- `configs/phase4/`: thresholding and visualization diagnostics.
- `configs/phase5/`: statistical baselines and score/component ablation.

## Agent Files Reviewed

| File | Status | Notes |
| --- | --- | --- |
| `AGENTS.md` | Current | Points agents to `run.py`, `main.py`, stage files, and YAML config policy. |
| `.codex/skills/gbm-anomaly-transformer/SKILL.md` | Current | Mirrors the lean YAML-driven workflow and warns against recreating one-off runners. |
| `README.md` | Current | Documents active stage files, general configs, phase configs, and output-root guidance. |
| `configs/README.md` | Current | Maps general and phase config folders. |

## Device Policy

Use `device: auto` or `DEVICE=auto` unless a specific accelerator is requested. Device resolution now lives in `src/gbm/train.py` and resolves:

```text
cuda -> mps -> cpu
```

Mac M1/M2/M3 work should use `environment.macos-mps.yml`. Keep generated artifacts outside the repo; the recommended Windows output root is:

```text
D:/AnomalyTransformerRuns
```

## No Longer Active As Primary Sources

These are retained only as historical context:

- Old `scripts/gbm/*` runner paths.
- Old `run_*.bat`, `run_*.sh`, and PowerShell experiment launchers.
- Legacy Anomaly Transformer / power-law implementation paths.
- Bulk generated old outputs, checkpoints, logs, and image-heavy report folders.

## Important Completed Evidence

| Evidence | Artifact |
| --- | --- |
| Dataset leakage and signal validation | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP1__insight.md` |
| Power-law prior diagnostic conclusion | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md` |
| GBM formulation plan | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP3__experiment_plan.md` |
| Score-change dynamic threshold legacy metrics | `docs/legacy/gbm-report-summaries/` |

## Still Unanswered Or Not Yet Publication-Ready

- Main benchmark against strong baselines with current cleaned code.
- Multi-seed evaluation with confidence intervals and significance tests.
- Robustness across regimes, time periods, stock subsets, and seeds.
- Formal early-warning lead-time or detection-delay analysis.
- Explanation faithfulness and explanation stability tests.
- Cross-sectional stock-relation contribution.
- Whether GBM distribution consistency beats the strongest baseline.

## Recommended Agent Behavior

- Use `run.py --config <yaml>` for all runnable workflows.
- Prefer `configs/general/` for routine runs and the phase folders for research-history reruns.
- Do not recreate deleted runner families unless the user explicitly reverses the cleanup decision.
- Treat current findings as partial evidence, not final paper claims.
- Use `docs/RESEARCH_QUESTION_ANSWER_MAP.md` before drafting research claims.
- Append concise implementation outcomes to `docs/research-plan-070662026.md`.
