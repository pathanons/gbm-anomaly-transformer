# General Run Configs

These configs are the day-to-day entry points for the canonical workflow.
Phase configs preserve experiment history; this folder is for routine reruns.

Run from the repo root:

```bash
python run.py --config configs/general/data_prepare.yaml --dry-run
python run.py --config configs/general/train.yaml --dry-run
python run.py --config configs/general/test.yaml --dry-run
python run.py --config configs/general/visualize.yaml --dry-run
```

- `data_prepare.yaml`: build/validate the joint manifest from an existing dataset folder.
- `train.yaml`: train the default Gaussian log-return attention model.
- `test.yaml`: test the trained default Gaussian log-return attention model.
- `visualize.yaml`: render final MAD threshold plots using the current k=9 visualization path.
