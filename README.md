# GBM Anomaly Transformer

GBM-aware Anomaly Transformer workspace for pooled, multi-ticker financial anomaly detection on S&P 500-style OHLCV windows.

The active path is the joint GBM pipeline:

```text
scripts/gbm/run_joint.py
  -> train_joint.py
  -> validate_joint.py
  -> test_joint.py
  -> visualize_joint.py optional
```

## Active Layout

```text
configs/                example configs
datasets/               prepared OHLCV and anomaly-label CSVs
scripts/gbm/            train, validate, test, visualize, score-change tools
scripts/data_prep/      event-taxonomy dataset generation utilities
src/gbm/                model, data loading, losses, metrics, scoring, device helpers
utils/                  shared dataset and validation helpers
docs/legacy/            curated old reports, insights, and research context
.github/                agent prompts and review instructions
.codex/skills/          repo-local Codex skill for this project
```

Generated outputs go under `results/experiments/<exp-name>/` and are ignored by git.

## Quick Run

Windows:

```bat
run.bat --visualize
```

macOS/Linux:

```bash
bash run.sh --visualize
```

Both scripts default to:

```text
EXP_NAME=experiment3_joint
DATA_PATH=datasets/SP500_event_taxonomy_w100
WINDOW_SIZE=100
FEATURES=all
BATCH_SIZE=32
EPOCHS=20
DEVICE=auto
```

Override defaults with environment variables or append any `run_joint.py` flags:

```bash
EXP_NAME=experiment3_mps DEVICE=mps EPOCHS=5 bash run.sh --visualize
```

```bat
set DEVICE=mps
set EPOCHS=5
run.bat --visualize
```

## Device Support

`DEVICE=auto` resolves in this order:

```text
cuda -> mps -> cpu
```

Use `--device mps` or `DEVICE=mps` on Mac M1/M2/M3. The entry points set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported MPS operations can fall back to CPU instead of crashing.

## Mac M1/M2/M3 Setup

Use the no-CUDA conda environment:

```bash
conda env create -f environment.macos-mps.yml
conda activate gbm-anomaly-transformer-mps
DEVICE=mps bash run.sh --visualize
```

For pip-based setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
DEVICE=mps bash run.sh --visualize
```

## Windows CUDA/CPU Setup

Use your existing PyTorch install if it already matches your CUDA version. For CPU-only smoke checks, set:

```bat
set DEVICE=cpu
run.bat --epochs 1
```

## Main Outputs

```text
results/experiments/experiment3_joint/models/
results/experiments/experiment3_joint/reports/
results/experiments/experiment3_joint/splits/
results/experiments/experiment3_joint/visualizations/
```

Key report files:

```text
gbm_joint_threshold.json
gbm_joint_validation_scores.csv
gbm_joint_test_scores.csv
gbm_joint_metrics.json
gbm_joint_metrics_by_ticker.csv
```

## Legacy Context

Only important old context was imported:

```text
docs/legacy/docs/
docs/legacy/experiment-insights/
docs/legacy/gbm-report-summaries/
docs/legacy/ticker-insight-json/
```

Large generated outputs, checkpoints, logs, and image-heavy result folders were intentionally left out.

## Agent And Codex Context

Repo-local guidance lives in:

```text
AGENTS.md
.codex/skills/gbm-anomaly-transformer/SKILL.md
.github/copilot-instructions.md
.github/agents/
.github/instructions/
.github/prompts/
```

Agents should treat `scripts/gbm/run_joint.py`, `src/gbm/device.py`, and this README as the current source of truth for running experiments.
