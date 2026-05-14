# Agent And Instruction Audit

Audit date: 2026-05-14

## Current Project State

The active repository is `gbm-anomaly-transformer`. The active research implementation is no longer the broad legacy Anomaly Transformer / power-law prototype. The current executable path is the GBM-aware joint model:

```text
scripts/gbm/run_joint.py
  -> scripts/gbm/train_joint.py
  -> scripts/gbm/validate_joint.py
  -> scripts/gbm/test_joint.py
  -> scripts/gbm/visualize_joint.py optional
```

The active device policy is:

```text
DEVICE=auto: cuda -> mps -> cpu
```

Mac M1/M2/M3 work should use `environment.macos-mps.yml` and `DEVICE=mps`.

## Agent Files Reviewed

| File | Status | Notes |
| --- | --- | --- |
| `AGENTS.md` | Current | Correctly points to active GBM runner, MPS support, and curated legacy context. |
| `.codex/skills/gbm-anomaly-transformer/SKILL.md` | Current | Correctly states current pipeline, device policy, and legacy-import guardrails. |
| `C:\Users\Acer\.codex\skills\gbm-anomaly-transformer\SKILL.md` | Current | Global Codex skill has been updated from the repo-local skill. |
| `.github/copilot-instructions.md` | Current | Updated for cross-platform execution and MPS. |
| `.github/agents/research-project.agent.md` | Mostly current | Still contains the full long-term publication roadmap; should be read as roadmap, not completed work. |
| `.github/agents/research-experiment-review.agent.md` | Current | Good for leakage, statistics, baselines, and MPS-aware run review. |
| `.github/agents/reviewer.agent.md` | Mostly current | Good alignment reviewer, but some roadmap items remain aspirational. |
| `.github/instructions/research-experiment-review.instructions.md` | Current | Apply target now points to README, AGENTS, legacy docs, and active GBM code. |
| `.github/prompts/research-experiment-review.prompt.md` | Current | Updated to use README/AGENTS/legacy context instead of missing old research docs. |

## No Longer Active As Primary Sources

These are retained only as historical context:

- Legacy Anomaly Transformer / power-law implementation path.
- Fixed gaussian-vs-power-law prior comparison as the final architecture direction.
- Old Windows-only batch logic from archived documents.
- Old references to `docs/research/RESEARCH_INTERVIEW.md`; that file is not present in this repo.
- Bulk generated old outputs, checkpoints, logs, and image-heavy report folders.

## What The Project Is Doing Now

The current project is testing a GBM-implied distribution consistency formulation:

- A Transformer encodes pooled multi-ticker windows.
- The model reconstructs input windows.
- The model predicts GBM-style return distribution parameters.
- The anomaly score combines reconstruction error, distributional likelihood, and divergence between observed and predicted return distribution.
- A score-change dynamic-threshold layer is being explored as an operational alerting rule.

## Important Completed Evidence

| Evidence | Artifact | Artifact date |
| --- | --- | --- |
| Dataset leakage and signal validation | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP1__insight.md` | 2026-04-13 23:00:09 +07:00 |
| Power-law prior diagnostic conclusion | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md` | 2026-04-15 14:22:12 +07:00 |
| GBM formulation plan | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP3__experiment_plan.md` | 2026-04-16 11:46:52 +07:00 |
| Score-change dynamic threshold metrics, strict rule | `docs/legacy/gbm-report-summaries/experiment4_score_change_dynamic_threshold/reports/score_change_dynamic_threshold_metrics.json` | 2026-05-04 20:51:06 +07:00 |
| Score-change dynamic threshold metrics, w20 k1.5 | `docs/legacy/gbm-report-summaries/experiment4_score_change_dynamic_threshold_w20_k1p5/reports/score_change_dynamic_threshold_metrics.json` | 2026-05-04 22:16:19 +07:00 |
| Score-change rule sweep | `docs/legacy/gbm-report-summaries/experiment4_score_change_rule_sweep/reports/score_change_rule_sweep_summary.csv` | 2026-05-04 22:15:01 +07:00 |

## Still Unanswered Or Not Yet Publication-Ready

- Main benchmark against vanilla Anomaly Transformer, LSTM/GRU autoencoder, forecasting baselines, and statistical baselines.
- Multi-seed evaluation with confidence intervals and significance tests.
- Robustness across regimes, time periods, stock subsets, and seeds for the current GBM model.
- Formal early-warning lead-time or detection-delay analysis.
- Explanation faithfulness and explanation stability tests.
- Cross-sectional stock-relation contribution.
- Whether GBM distribution consistency beats the strongest baseline.
- Whether score-change dynamic thresholding improves operational early warning without unacceptable recall loss.

## Recommended Agent Behavior

- Treat current findings as partial evidence, not final paper claims.
- Use `docs/RESEARCH_QUESTION_ANSWER_MAP.md` before drafting any claims.
- Do not state that the proposed model is superior until benchmark, multi-seed, and statistical tests exist.
- Do not state interpretability claims until faithfulness/stability checks exist.
- Do not state early-warning superiority until lead-time metrics exist.
