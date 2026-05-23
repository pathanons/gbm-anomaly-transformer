# Financial Prior Attention Transformers

Workspace for pooled, multi-ticker financial anomaly detection on S&P 500-style OHLCV windows. There are two distinct attention-prior models:

- Gaussian log-return attention: a Transformer with a Gaussian prior on log-return transition likelihoods.
- Canonical GBM attention: a Transformer whose attention prior uses the canonical GBM price process and applies the Ito correction inside the log-price transition law.

Use the dedicated runners below so these two models are not mixed accidentally:

```text
scripts/gbm/run_gaussian_log_return_attention.py
scripts/gbm/run_canonical_gbm_attention.py
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

Generated outputs go under the configured output root, then `experiments/<exp-name>/`. By default this is `results/experiments/<exp-name>/`; set `AT_OUTPUT_ROOT` to move generated runs elsewhere. On Windows, `run.bat` and `train.bat` default to `D:\AnomalyTransformerRuns` to keep large outputs off `C:/`.

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
PREDICTIVE_DISTRIBUTION=gaussian
THRESHOLD_METHOD=quantile
AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns on Windows .bat entrypoints
```

Override defaults with environment variables or append any `run_joint.py` flags. For example, choose a different output drive/root:

```bash
AT_OUTPUT_ROOT=/Volumes/ResearchRuns/AnomalyTransformer EXP_NAME=experiment3_mps DEVICE=mps EPOCHS=5 bash run.sh --visualize
```

```bat
set AT_OUTPUT_ROOT=D:\AnomalyTransformerRuns
set DEVICE=cpu
set EPOCHS=5
run.bat --visualize
```

Gaussian log-return attention:

```bat
run_log_return_attention.bat --predictive-distribution student_t --threshold-method conformal --threshold-quantile 0.95
```

```bash
bash run_log_return_attention.sh --predictive-distribution student_t --threshold-method conformal --threshold-quantile 0.95
```

Canonical GBM attention:

```bat
run_canonical_gbm_attention.bat --predictive-distribution student_t --threshold-method conformal --threshold-quantile 0.95
```

```bash
bash run_canonical_gbm_attention.sh --predictive-distribution student_t --threshold-method conformal --threshold-quantile 0.95
```

By default, the joint data loader uses chronological splits with an embargo gap derived from the window size, fits normalization on training windows only, and trains only on normal training windows. Use `--include-anomalous-train` only for contamination ablations.

Use `--association-mode none` for the no-prior Transformer ablation and `--association-mode temporal` for a local temporal-prior ablation.

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
%AT_OUTPUT_ROOT%/experiments/experiment3_joint/models/
%AT_OUTPUT_ROOT%/experiments/experiment3_joint/reports/
%AT_OUTPUT_ROOT%/experiments/experiment3_joint/splits/
%AT_OUTPUT_ROOT%/experiments/experiment3_joint/visualizations/
```

Key report files:

```text
gbm_joint_threshold.json
gbm_joint_validation_scores.csv
gbm_joint_test_scores.csv
gbm_joint_metrics.json
gbm_joint_metrics_by_ticker.csv
gbm_joint_metrics_by_event_type.csv
```

`validate_joint.py` fits the anomaly threshold on validation scores only. The default decision rule is `score > validation normal-score quantile`; `--threshold-method conformal` instead stores validation calibration scores and uses conformal p-values in test. `test_joint.py` then locks the validation rule and reports point metrics, tolerance-window metrics, by-ticker metrics, and by-event-type catch rates.

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
