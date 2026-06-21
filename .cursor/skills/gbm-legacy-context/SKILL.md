---
name: gbm-legacy-context
description: Navigates curated legacy research context imported from the old Anomaly-Transformer repo. Use when answering research-history questions, citing prior EXP1–EXP4 evidence, interpreting event labels, reviewing old experiment plans, or deciding what historical claims are safe to reuse in gbm-anomaly-transformer.
---

# GBM Legacy Research Context

Use this skill when the task depends on **historical evidence, old experiment notes, or prior research direction** — not when implementing the active GBM pipeline (use the main `gbm-anomaly-transformer` skill for that).

## Source Of Truth Hierarchy

Read in this order:

1. **Active code** — `scripts/gbm/`, `src/gbm/`
2. **Current research map** — `docs/RESEARCH_QUESTION_ANSWER_MAP.md`
3. **Agent audit** — `docs/AGENT_CONTEXT_AUDIT.md`
4. **Curated legacy** — `docs/legacy/` (this skill)

Do not treat legacy docs as describing the current executable path unless confirmed against active code.

## What `docs/legacy/` Contains

| Subfolder | Purpose |
| --- | --- |
| `docs/legacy/docs/` | Old research guides, summaries, event-label spec, archived training notes |
| `docs/legacy/experiment-insights/` | Human-written EXP1–EXP3 plans, checklists, and insight write-ups |
| `docs/legacy/gbm-report-summaries/` | Textual summaries from score-change / dynamic-threshold sweeps (May 2026) |
| `docs/legacy/ticker-insight-json/` | Per-ticker statistical insight JSON (no bulk run artifacts) |

See [reference.md](reference.md) for the key file map.

## Evidence Index (Safe To Cite)

| ID | Topic | Path |
| --- | --- | --- |
| E1 | Dataset validation, leakage checks, label separability | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP1__insight.md` |
| E2 | Power-law prior diagnostic (deprioritized as final design) | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP2__insight_results.md` |
| E3 | GBM distribution-consistency experiment plan | `docs/legacy/experiment-insights/old__inactive_after_gbm_joint__TODO__EXP3__experiment_plan.md` |
| E4–E7 | Score-change dynamic threshold / rule-sweep summaries | `docs/legacy/gbm-report-summaries/experiment4_*` |

Always pair citations with artifact dates from `docs/RESEARCH_QUESTION_ANSWER_MAP.md`.

## What Is Historical (Do Not Re-Activate Blindly)

- Fixed **power-law attention prior** as the main contribution (E2 says geometry changes, robust detection gains are weak).
- Old **Anomaly Transformer / power-law prototype** paths referenced in archived docs.
- **Score-change / calibration sweep scripts** that existed in prior repo iterations — summaries live under `gbm-report-summaries/`; reimplementation requires checking current `scripts/gbm/` first.
- File names prefixed `old__inactive_after_gbm_joint__` — retained for traceability, not as live run configs.
- Old doc index links under `docs/legacy/docs/old__docs__INDEX.md` — many linked paths do not exist in this repo.

## Event Taxonomy Quick Reference

For label semantics and window parameters, read:

`docs/legacy/docs/old__inactive_after_gbm_joint__docs__research__EVENT_LABEL_SPEC.md`

Event types: `jump`, `drop`, `volume_spike`, `volatility_shock`, `regime_shift`. Default dataset: `SP500_event_taxonomy_w100`.

## Research Guardrails

- Legacy evidence supports **partial answers** only. Check `docs/RESEARCH_QUESTION_ANSWER_MAP.md` for open vs answered RQs.
- Do **not** claim model superiority, early-warning superiority, or explanation faithfulness from legacy artifacts alone.
- Do **not** bulk-import old checkpoints, logs, image reports, or per-run diagnostics into this repo.
- Validation-only tuning remains mandatory; never tune on test results.
- When drafting paper claims, prefer E1 (dataset validity) and E2 (why power-law was deprioritized) plus any **new** runs in the active pipeline.

## Common Tasks

### Answer “what did we learn from EXP1/2/3?”

1. Read the matching file under `docs/legacy/experiment-insights/`.
2. Cross-check status in `docs/RESEARCH_QUESTION_ANSWER_MAP.md`.
3. State limits explicitly (multi-seed, baselines, lead-time often still open).

### Answer “what happened with score-change thresholds?”

1. Read manifests and CSV/JSON under `docs/legacy/gbm-report-summaries/experiment4_*`.
2. Treat as **exploratory evidence**, not locked production behavior.
3. Confirm whether active code still implements the rule before recommending it.

### Find ticker-level historical stats

Search `docs/legacy/ticker-insight-json/{TICKER}/{TICKER}_insights.json`.

### Distinguish roadmap from completed work

`.github/agents/research-project.agent.md` is a long-term roadmap. Legacy experiment checklists may have unchecked boxes — verify against active code and fresh runs.

## Import Policy

`docs/legacy/README.md` defines what was kept vs excluded. Add only **small curated summaries** to git; keep large generated outputs outside the repo.
