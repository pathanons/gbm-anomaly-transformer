---
name: gbm-anomaly-transformer
description: Use when working in this repo on Gaussian log-return or canonical GBM attention experiments, cross-platform runners, Mac Apple Silicon MPS support, validation/test evaluation, or curated legacy research context.
---

# GBM Anomaly Transformer Skill

Use this skill for the active `gbm-anomaly-transformer` repo.

## Current Pipeline Shape

This repo is now intentionally lean and YAML-driven. The active executable path is:

```text
run.py -> main.py -> src/gbm/<stage>.py
```

The active stage files are:

- `src/gbm/datasets.py` for data preparation and window manifests.
- `src/gbm/model.py` for model definitions.
- `src/gbm/score.py` for loss and score definitions.
- `src/gbm/train.py` for training.
- `src/gbm/test.py` for validation/testing and score export.
- `src/gbm/visualize.py` for configurable k*MAD visualization.
- `src/gbm/statistics.py` for statistical baselines and evaluation.

Do not add new one-off `run_*.py`, `*_joint.py`, `visualize_*.py`, or experiment-specific Python entry points. Add experiment variants as YAML configs and reusable options in the existing stage files.

Do not default to archived paths from the old `Anomaly-Transformer` repo.

Canonical GBM attention is implemented as an explicit latent source timestamp
posterior over past timestamps, using GBM log-price transition likelihoods with
Ito correction and per-step date deltas. Do not describe it as a standalone
Bayesian prior emitted by GBM without mentioning the latent timestamp model.

## Config Layout

- `configs/general/`: routine data preparation, train, test, configurable MAD visualization, and the best-model suite.
- `configs/phase1/`: dataset preparation, statistical evaluation, data insight, legacy/refactored score checks, and loss ablation.
- `configs/phase2/`: distribution-shift score variants (`legacy`, `refactored`, `qw2`, `qw2_tail`).
- `configs/phase3/`: model and association-mode experiments (`gaussian_log_return`, `canonical_gbm`, `temporal`, `none`).
- `configs/phase4/`: thresholding and visualization diagnostics.
- `configs/phase5/`: baseline evaluation and score/component ablation.

## Device Policy

Use `DEVICE=auto` by default. Device resolution now lives in `src/gbm/train.py` and resolves:

```text
cuda -> mps -> cpu
```

For Mac M1/M2/M3, create the environment with:

```bash
conda env create -f environment.macos-mps.yml
conda activate gbm-anomaly-transformer-mps
```

Use `device: auto` in YAML unless the user asks for a specific accelerator.

## Run Commands

General routine configs:

```bash
python run.py --config configs/general/data_prepare.yaml --dry-run
python run.py --config configs/general/train.yaml --dry-run
python run.py --config configs/general/test.yaml --dry-run
python run.py --config configs/general/visualize.yaml --dry-run
python run.py --config configs/general/best_model_suite.yaml --dry-run
```

Research phase example:

```bash
python run.py --config configs/phase3/example_log_return.yaml --dry-run
```

Set `AT_OUTPUT_ROOT=D:/AnomalyTransformerRuns` for training/testing so artifacts resolve outside the repo.

Fast pre-run unit tests:

```powershell
python -m pytest unittest
```

Some tests use `pytest.importorskip("torch")` so lightweight Python
interpreters without PyTorch can still run the non-torch config and
visualization checks.

Best-current-model long suite:

```powershell
best.bat
```

Manual equivalent:

```powershell
$env:AT_OUTPUT_ROOT="D:/AnomalyTransformerRuns"
python run.py --config configs/general/best_model_suite.yaml
```

Use `best.bat --test-only` for the pytest gate and `best.bat --dry-run` for
pytest plus suite expansion without starting training.

This expands the current Gaussian log-return / NLL+association setup across
heads, layers, learning rates, epochs, and seeds, then runs MAD threshold
visualization for k values from 9 upward as configured in YAML. Treat the seed
sweep as a stability/CV proxy until true walk-forward cross-validation is
implemented.

## Research Guardrails

- Validation only for thresholds, hyperparameters, and model selection.
- Test set only for final evaluation.
- No improvement claims without statistical evidence.
- Report uncertainty for repeated runs.
- Keep generated outputs out of git unless they are small curated summaries.
- Read `docs/RESEARCH_QUESTION_ANSWER_MAP.md` before answering research-status or paper-claim questions.
- Treat baseline superiority, early-warning lead time, explanation faithfulness, and cross-sectional contribution as open questions unless newer evidence is present.

## Legacy Context

Use `docs/legacy/` for old insights and reports. It is intentionally curated. Do not bulk-copy raw old results, checkpoints, logs, or image-heavy outputs back into this repo.
