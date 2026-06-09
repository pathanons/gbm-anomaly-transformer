# Config Phase Map

Run every config from the repo root:

```bash
python run.py --config configs/phase1/data_prepare.yaml --dry-run
```

- `general/`: day-to-day workflow configs for data preparation, train, test, and k=9 MAD visualization.
- `phase1/`: dataset preparation, data statistics, legacy/refactored loss-score checks, and loss ablation.
- `phase2/`: distribution-shift score study (`legacy`, `refactored`, `qw2`, `qw2_tail`).
- `phase3/`: model and association-mode experiments (`log_return`, `canonical`, `temporal`, `none`).
- `phase4/`: thresholding and visualization diagnostics.
- `phase5/`: baseline evaluation, score/component ablation, and final comparison setup.

Legacy-only experiments are documented in each phase README when the original code path was intentionally removed during repo cleanup.
