from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from src.gbm.test import _select_extreme_and_normal_context


def test_extreme_normal_context_selector_keeps_reference_lags(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=16, freq="D"),
            "Close": [100, 101, 102, 103, 104, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140],
            "Volume": [1000] * 16,
        }
    ).to_csv(data_dir / "AAA_ohlcv.csv", index=False)

    scores = pd.DataFrame(
        {
            "ticker": ["AAA"] * 12,
            "split": ["test"] * 12,
            "window_id": list(range(100, 112)),
            "end_date": pd.date_range("2026-01-05", periods=12, freq="D"),
            "score": list(range(12)),
        }
    )
    args = type("Args", (), {"data_path": str(data_dir), "attention_reference_count": 1, "attention_context_days": 2})()

    selected = _select_extreme_and_normal_context(scores, args)

    assert {"max_abs_log_return", "normal_near_mean"} == set(selected["reference_role"])
    assert set(selected["lag_from_reference_days"]).issubset({0, 1, 2})
    assert (selected["lag_from_reference_days"] == 0).any()
