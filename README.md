# Anomaly Transformer GBM Workspace

Active code path: GBM-aware Transformer trained on pooled multi-ticker windows.

## Active Layout

```text
scripts/gbm/          GBM train, validate, test, visualize, joint runner
scripts/data_prep/    dataset generation utilities
src/gbm/              GBM model, data, losses, scoring, metrics
src/gbm/embed.py      embedding layer used by the GBM model
utils/                shared dataset and label helpers
```

Legacy Anomaly Transformer / power-law / regime code has been moved to:

```text
old/inactive_after_gbm_joint/
```

## Main Model

```text
src/gbm/model.py
```

The model reconstructs each input window and predicts GBM-style return distribution parameters.

## Main Training Path

Run the pooled joint GBM pipeline:

```powershell
python scripts/gbm/run_joint.py --exp-name experiment3_joint --data-path datasets/SP500_event_taxonomy_w100 --window-size 100 --features all --batch-size 32 --epochs 20 --device cuda
```

This runs:

```text
train_joint.py -> validate_joint.py -> test_joint.py
```

Add `--visualize` to also run `visualize_joint.py`.

## Main Outputs

```text
results/experiments/experiment3_joint/models/
results/experiments/experiment3_joint/reports/
results/experiments/experiment3_joint/splits/
results/experiments/experiment3_joint/visualizations/
```

## Notes

- `results/`, `logs/`, generated figures, and checkpoints are ignored by git.
- The active model path is GBM joint pooled training.
- Archived code is preserved under `old/inactive_after_gbm_joint/` but is no longer part of the active workflow.
