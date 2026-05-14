---
description: "Workspace-wide agent instructions for gbm-anomaly-transformer"
---

This repository is the active `gbm-anomaly-transformer` workspace. The current source of truth is the pooled joint GBM pipeline, not archived legacy paths.

## Active Run Path

- Main runner: `scripts/gbm/run_joint.py`
- Windows wrapper: `run.bat`
- macOS/Linux wrapper: `run.sh`
- Device helper: `src/gbm/device.py`
- Mac Apple Silicon environment: `environment.macos-mps.yml`

Use `DEVICE=auto` unless the user asks for a specific target. Auto resolves to CUDA, then Apple Silicon MPS, then CPU. For Mac M1/M2/M3, prefer `DEVICE=mps bash run.sh`.

## Python Execution

Do not assume a single global interpreter. Prefer the environment the user has activated. On Mac M1/M2/M3, use the `gbm-anomaly-transformer-mps` conda environment from `environment.macos-mps.yml`. On Windows, use the repo's active environment or the user's requested Conda environment.

## Research Constraints

- Validation data only for thresholds, hyperparameters, and model selection.
- Test data only for final evaluation.
- No claims of improvement without statistical support.
- Keep generated checkpoints, logs, images, and bulk result artifacts out of git.
- Use `docs/legacy/` only as curated historical context.
