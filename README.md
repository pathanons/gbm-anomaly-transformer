# GBM Anomaly Transformer

This repo now uses one YAML-driven entry point:

```bash
python run.py --config configs/phase3/example_log_return.yaml
```

Use `--dry-run` to print the resolved experiment without training:

```bash
python run.py --config configs/phase3/example_log_return.yaml --dry-run
```

## Active Python Surface

The active code is organized by pipeline stage:

- `main.py` - config dispatcher and experiment orchestration
- `run.py` - thin executable entry point
- `src/gbm/datasets.py` - data preparation and windows
- `src/gbm/model.py` - model definitions
- `src/gbm/score.py` - loss and score definitions
- `src/gbm/train.py` - training
- `src/gbm/test.py` - validation, testing, and score export
- `src/gbm/visualize.py` - plots and visual diagnostics
- `src/gbm/statistics.py` - statistics, baselines, and evaluation

Experiment choices live in YAML, not in separate runner filenames.

## General Configs

Use these for normal reruns:

```bash
python run.py --config configs/general/data_prepare.yaml
python run.py --config configs/general/train.yaml
python run.py --config configs/general/test.yaml
python run.py --config configs/general/visualize.yaml
python run.py --config configs/general/best_model_suite.yaml --dry-run
```

Run the fast pytest gate before long experiments:

```powershell
python -m pytest unittest
```

On Windows, the curated best-model launcher is:

```powershell
best.bat
```

Useful checks:

```powershell
best.bat --test-only
best.bat --dry-run
```

## Configs By Phase

- `configs/general/` - routine data preparation, train, test, configurable MAD visualization, and best-model suite configs
- `configs/phase1/` - data preparation, data statistics, legacy/refactored score checks, loss ablation
- `configs/phase2/` - distribution-shift score modes: legacy, refactored, QW2, QW2Tail
- `configs/phase3/` - log-return/canonical model configs and association-mode ablations
- `configs/phase4/` - MAD threshold visualization diagnostics
- `configs/phase5/` - statistical baselines and score/component ablation

Large outputs should go outside the repo. Recommended Windows output root:

```powershell
$env:AT_OUTPUT_ROOT = "D:/AnomalyTransformerRuns"
python run.py --config configs/phase3/example_log_return.yaml
```
