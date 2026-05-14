---
name: gbm-anomaly-transformer
description: Use when working in this repo on GBM-aware Anomaly Transformer experiments, cross-platform runners, Mac Apple Silicon MPS support, validation/test evaluation, or curated legacy research context.
---

# GBM Anomaly Transformer Skill

Use this skill for the active `gbm-anomaly-transformer` repo.

## Current Pipeline

The active experiment runner is:

```bash
python scripts/gbm/run_joint.py
```

It orchestrates:

```text
train_joint.py -> validate_joint.py -> test_joint.py -> visualize_joint.py optional
```

Do not default to archived paths from the old `Anomaly-Transformer` repo.

## Device Policy

Use `DEVICE=auto` by default. The helper in `src/gbm/device.py` resolves:

```text
cuda -> mps -> cpu
```

For Mac M1/M2/M3, use:

```bash
conda env create -f environment.macos-mps.yml
conda activate gbm-anomaly-transformer-mps
DEVICE=mps bash run.sh --visualize
```

The Python entry points and runners set `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon compatibility.

## Run Commands

Windows:

```bat
run.bat --visualize
```

macOS/Linux:

```bash
bash run.sh --visualize
```

Override defaults with `EXP_NAME`, `DATA_PATH`, `WINDOW_SIZE`, `FEATURES`, `BATCH_SIZE`, `EPOCHS`, and `DEVICE`.

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
