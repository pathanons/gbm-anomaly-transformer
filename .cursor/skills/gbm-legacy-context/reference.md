# Legacy File Map

## Entry Points

| File | Use when |
| --- | --- |
| `docs/legacy/README.md` | Understanding import scope and exclusions |
| `docs/legacy/docs/old__docs__INDEX.md` | Browsing old doc taxonomy (links may be stale) |
| `docs/RESEARCH_QUESTION_ANSWER_MAP.md` | Mapping RQs to evidence IDs |
| `docs/AGENT_CONTEXT_AUDIT.md` | Agent/instruction freshness audit |

## Experiment Insights (`docs/legacy/experiment-insights/`)

| File | Content |
| --- | --- |
| `old__inactive_after_gbm_joint__TODO__EXP1__insight.md` | E1 — dataset validation conclusions |
| `old__inactive_after_gbm_joint__TODO__EXP1__dataset_validation.md` | EXP1 validation checklist (mostly checked) |
| `old__inactive_after_gbm_joint__TODO__EXP2__experiment_plan.md` | Power-law prior experiment design |
| `old__inactive_after_gbm_joint__TODO__EXP2__checklist.md` | EXP2 execution checklist |
| `old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md` | E2 — power-law diagnostic results |
| `old__inactive_after_gbm_joint__TODO__EXP3__experiment_plan.md` | E3 — GBM joint model plan |
| `old__inactive_after_gbm_joint__TODO__EXP3__checklist.md` | EXP3 checklist (may be partially stale) |

## Research Docs (`docs/legacy/docs/`)

| File | Content |
| --- | --- |
| `old__inactive_after_gbm_joint__docs__research__EVENT_LABEL_SPEC.md` | Event taxonomy and labeling rules |
| `old__inactive_after_gbm_joint__docs__research__RESEARCH_FRAMEWORK.md` | Original H1–H4 framework (power-law era) |
| `old__inactive_after_gbm_joint__docs__research__RESEARCH_OUTLINE.md` | High-level outline |
| `old__inactive_after_gbm_joint__docs__research__RESEARCH_JOURNEY.md` | Progress log |
| `old__inactive_after_gbm_joint__docs__research__GOAL_AIM.md` | Goal statement |
| `old__docs__summaries__EXECUTIVE_SUMMARY.md` | Old executive snapshot |
| `old__docs__summaries__VISUAL_OVERVIEW.md` | ASCII architecture diagrams |
| `old__docs__archived__TRAINING_GUIDE.md` | Archived training procedures |
| `old__docs__archived__QUICK_REFERENCE.md` | Old command cheatsheet |
| `old__docs__archived__PROJECT_STATUS_REPORT.md` | Historical status |
| `old__DEEP_ANALYSIS_EXECUTION_GUIDE.md` | Deep analysis workflow |

## GBM Report Summaries (`docs/legacy/gbm-report-summaries/`)

| Folder | Reports |
| --- | --- |
| `experiment4_score_change_dynamic_threshold/` | Strict rolling-MAD score-change rule metrics |
| `experiment4_score_change_dynamic_threshold_w20_k1p5/` | w=20, k=1.5 variant |
| `experiment4_score_change_rule_sweep/` | Rule sweep summary, by-event-type, best configs |

Each folder has a `*_manifest.json` describing source experiment and parameters.

## Ticker Insights

Pattern: `docs/legacy/ticker-insight-json/{TICKER}/{TICKER}_insights.json`

Statistical summaries per S&P 500 ticker from prior EDA — useful for cross-sectional context, not model checkpoints.

## Filename Prefix Legend

| Prefix | Meaning |
| --- | --- |
| `old__inactive_after_gbm_joint__` | Superseded after GBM joint pivot; keep for evidence chain |
| `old__docs__` | Imported from old repo `docs/` tree |
| `old__docs__archived__` | Explicitly archived in source repo |
