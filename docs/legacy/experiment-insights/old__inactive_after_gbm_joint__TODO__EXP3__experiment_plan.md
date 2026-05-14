# EXP3: GBM-Implied Distribution Consistency

Goal: replace the prior-series discrepancy story with a GBM-implied distribution consistency detector for daily equity data.

Core idea:
- Transformer encodes each stock window.
- One head predicts a GBM-implied return distribution.
- Observed window returns provide the empirical distribution.
- Anomaly score is the divergence between the two distributions plus reconstruction error.

Primary outputs:
- train/validate/test pipeline
- attention export
- per-window divergence and anomaly scores
- publication-ready summary tables
