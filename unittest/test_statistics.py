from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from src.gbm.statistics import parse_baselines, score_manifest, summarize_baseline


def test_parse_baselines_validates_names() -> None:
    assert parse_baselines("rolling_volatility,last_return_zscore") == ["rolling_volatility", "last_return_zscore"]
    assert "rolling_volatility" in parse_baselines("all")
    with pytest.raises(ValueError):
        parse_baselines("missing_baseline")


def test_score_manifest_and_summary_use_validation_threshold() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "ticker": ["AAA"] * 4,
            "split": ["train", "val", "test", "test"],
            "start_idx": [0, 1, 2, 3],
            "end_idx": [3, 4, 5, 6],
            "start_date": pd.date_range("2026-01-01", periods=4),
            "end_date": pd.date_range("2026-01-03", periods=4),
            "y_true": [0, 0, 0, 1],
        }
    )
    window_store = {"AAA": {"returns": np.array([0.0, 0.01, 0.02, 0.05, 0.2, 0.3])}}

    scores = score_manifest(manifest, window_store, "mean_abs_return", lambda returns: float(np.mean(np.abs(returns))))
    summary = summarize_baseline(scores, threshold_quantile=0.5)

    assert len(scores) == 4
    assert summary["baseline"] == "mean_abs_return"
    assert summary["val_windows"] == 1
    assert summary["test_windows"] == 2
