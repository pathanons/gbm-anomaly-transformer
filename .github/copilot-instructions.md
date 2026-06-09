---
description: "Workspace-wide agent instructions for gbm-anomaly-transformer"
---

This repository is the active `gbm-anomaly-transformer` workspace. The current source of truth is the lean YAML-driven GBM pipeline, not archived legacy paths or deleted one-off runners.

## Active Run Path

- Single executable entry point: `run.py`
- Single orchestration entry point: `main.py`
- General configs: `configs/general/*.yaml`
- Phase experiment configs: `configs/phase1` through `configs/phase5`
- Device resolution: `src/gbm/train.py`
- Mac Apple Silicon environment: `environment.macos-mps.yml`

Use `device: auto` in YAML unless the user asks for a specific target. Auto resolves to CUDA, then Apple Silicon MPS, then CPU.

Routine examples:

```bash
python run.py --config configs/general/data_prepare.yaml --dry-run
python run.py --config configs/general/train.yaml --dry-run
python run.py --config configs/general/test.yaml --dry-run
python run.py --config configs/general/visualize.yaml --dry-run
```

## Python Execution

Do not assume a single global interpreter. Prefer the environment the user has activated. On Mac M1/M2/M3, use the `gbm-anomaly-transformer-mps` conda environment from `environment.macos-mps.yml`. On Windows, use the repo's active environment or the user's requested Conda environment.

Do not add new one-off `run_*.py`, `*_joint.py`, `visualize_*.py`, or experiment-specific Python entry points. Add variants as YAML configs and reusable options in the existing stage files.

## Research Constraints

- Validation data only for thresholds, hyperparameters, and model selection.
- Test data only for final evaluation.
- No claims of improvement without statistical support.
- Keep generated checkpoints, logs, images, and bulk result artifacts out of git.
- Use `docs/legacy/` only as curated historical context.
