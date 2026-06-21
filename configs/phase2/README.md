# Phase 2: Distribution-Shift Score Study

Purpose:
- Compare score modes over the same GBM log-return model family.
- Reproduce the `legacy`, `refactored`, `qw2`, and `qw2_tail` score-mode checks.

Runnable configs:
- `score_legacy.yaml`
- `score_refactored.yaml`
- `score_qw2.yaml`
- `score_qw2_tail.yaml`

Known result from prior run:
- QW2 and QW2Tail were nearly tied with legacy globally and did not provide a statistically meaningful gain.
