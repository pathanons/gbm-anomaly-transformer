# Agent Instructions

This repo is the active `gbm-anomaly-transformer` workspace. Keep the active code path lean and YAML-driven.

## Source Of Truth

- Single run entry point: `run.py`
- Single orchestration entry point: `main.py`
- Data preparation: `src/gbm/datasets.py`
- Model definitions: `src/gbm/model.py`
- Loss and scoring definitions: `src/gbm/score.py`
- Training: `src/gbm/train.py`
- Validation/testing: `src/gbm/test.py`
- Visualization: `src/gbm/visualize.py`
- Statistics/evaluation/baselines: `src/gbm/statistics.py`
- Active research plan: `docs/research-plan-070662026.md`

Do not add new one-off `run_*.py`, `*_joint.py`, `visualize_*.py`, or experiment-specific Python entry points. Add experiment variants as YAML configs and options in `main.py` instead.

## Running

Use `DEVICE=auto` unless a user asks for a specific accelerator. `auto` resolves to CUDA, then Apple Silicon MPS, then CPU.

General routine configs:

```bash
python run.py --config configs/general/data_prepare.yaml
python run.py --config configs/general/train.yaml
python run.py --config configs/general/test.yaml
python run.py --config configs/general/visualize.yaml
```

Windows/macOS/Linux:

```bash
python run.py --config configs/phase3/example_log_return.yaml
```

Dry-run:

```bash
python run.py --config configs/phase3/example_log_return.yaml --dry-run
```

## Output Location

- Default experiment artifact location for this workspace is `D:/AnomalyTransformerRuns/experiments`.
- When running training/validation/test configs, set `AT_OUTPUT_ROOT=D:/AnomalyTransformerRuns` so outputs resolve to `D:/AnomalyTransformerRuns/experiments/<exp_name>`.
- Keep generated results, checkpoints, logs, and figures out of git unless the user explicitly asks to preserve a small report artifact.

## Research Rules

- Use validation data only for thresholds, hyperparameters, and model selection.
- Do not tune on test results.
- Do not claim improvement without statistical support.
- Report variance, confidence intervals, and significance tests when comparing repeated runs or baselines.
- Check `docs/RESEARCH_QUESTION_ANSWER_MAP.md` before making research claims.
- Treat baseline superiority, early-warning lead time, explanation faithfulness, and cross-sectional contribution as open until new evidence is added.
- Before implementation work, read `docs/research-plan-070662026.md` and keep it updated with concise implementation outcomes, validation findings, and open issues.
- When a task completes, append a short result note to `docs/research-plan-070662026.md`.

## Config Layout

- `configs/general/` holds day-to-day data preparation, train, test, and k=9 MAD visualization configs.
- `configs/phase1/` through `configs/phase5/` preserve the research experiment history by phase.
- Add new experiments as YAML configs in the closest phase folder. Add new reusable behavior to the existing stage files instead of creating separate runner scripts.
