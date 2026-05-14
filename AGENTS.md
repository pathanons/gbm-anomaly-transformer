# Agent Instructions

This repo is the active `gbm-anomaly-transformer` workspace. Prefer the current GBM joint pipeline over archived Anomaly Transformer, power-law, or regime code.

## Source Of Truth

- Runtime entry point: `scripts/gbm/run_joint.py`
- Device helper: `src/gbm/device.py`
- Cross-platform launchers: `run.bat`, `run.sh`
- Mac MPS environment: `environment.macos-mps.yml`
- Curated legacy context: `docs/legacy/README.md`
- Research claim map: `docs/RESEARCH_QUESTION_ANSWER_MAP.md`
- Agent/context audit: `docs/AGENT_CONTEXT_AUDIT.md`

## Running

Use `DEVICE=auto` unless a user asks for a specific accelerator. `auto` resolves to CUDA, then Apple Silicon MPS, then CPU.

Mac M1/M2/M3:

```bash
conda env create -f environment.macos-mps.yml
conda activate gbm-anomaly-transformer-mps
DEVICE=mps bash run.sh --visualize
```

Windows:

```bat
run.bat --visualize
```

## Research Rules

- Use validation data only for thresholds, hyperparameters, and model selection.
- Do not tune on test results.
- Do not claim improvement without statistical support.
- Report variance, confidence intervals, and significance tests when comparing repeated runs or baselines.
- Keep generated results, checkpoints, logs, and figures out of git unless the user explicitly asks to preserve a small report artifact.
- Check `docs/RESEARCH_QUESTION_ANSWER_MAP.md` before making research claims.
- Treat baseline superiority, early-warning lead time, explanation faithfulness, and cross-sectional contribution as open until new evidence is added.

## Editing Rules

- Keep active GBM code under `scripts/gbm/` and `src/gbm/`.
- Keep legacy imports curated under `docs/legacy/`.
- Do not re-import bulk old outputs from the old repo.
- When adding run instructions, include Windows and macOS/Linux forms.
