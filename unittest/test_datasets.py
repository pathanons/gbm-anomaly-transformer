from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from src.gbm.datasets import (
    JointWindowDataset,
    add_derived_features,
    build_windows_for_ticker,
    create_joint_manifest,
    discover_tickers,
    get_feature_columns,
)


def make_ticker_files(root: Path, ticker: str = "AAA", rows: int = 40) -> None:
    dates = pd.date_range("2026-01-01", periods=rows, freq="D")
    prices = np.linspace(100.0, 120.0, rows)
    ohlcv = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices + np.sin(np.arange(rows)) * 0.5,
            "Volume": np.linspace(1000, 2000, rows),
        }
    )
    labels = pd.DataFrame(
        {
            "Date": dates,
            "jump": [0] * rows,
            "drop": [0] * rows,
            "volatility_shock": [0] * rows,
        }
    )
    labels.loc[rows // 2, "jump"] = 1
    ohlcv.to_csv(root / f"{ticker}_ohlcv.csv", index=False)
    labels.to_csv(root / f"{ticker}_anomaly_label.csv", index=False)


def test_add_derived_features_and_feature_sets() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=3),
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [10.0, 11.0, 12.0],
            "Volume": [100, 150, 200],
        }
    )

    enriched = add_derived_features(frame)

    assert {"LogReturn", "LogVolume", "RealizedVol"}.issubset(enriched.columns)
    assert get_feature_columns("price_only") == ["Open", "High", "Low", "Close", "LogReturn", "RealizedVol"]


def test_create_joint_manifest_and_dataset_shapes(tmp_path: Path) -> None:
    make_ticker_files(tmp_path)

    manifest, window_store, scaler = create_joint_manifest(
        data_path=str(tmp_path),
        tickers=["AAA"],
        window_size=8,
        step=2,
        features="all",
        seed=42,
        split_method="chronological",
        purge_gap=0,
        train_normal_only=False,
    )
    dataset = JointWindowDataset(
        manifest,
        window_store,
        scaler,
        window_size=8,
        normalize_batch=False,
        split="train",
        train_normal_only=False,
    )
    item = dataset[0]

    assert set(manifest["split"]).issuperset({"train", "val", "test"})
    assert item["x"].shape == (8, len(get_feature_columns("all")))
    assert item["returns"].shape == (8,)
    assert item["time_deltas"].shape == (8,)
    assert item["meta"]["ticker"] == "AAA"
    assert "target_return" not in item


def test_next_day_target_uses_following_return(tmp_path: Path) -> None:
    make_ticker_files(tmp_path, rows=12)

    manifest, window_store, scaler = create_joint_manifest(
        data_path=str(tmp_path),
        tickers=["AAA"],
        window_size=4,
        step=1,
        features="log_return_tail_vol",
        seed=42,
        split_method="chronological",
        purge_gap=0,
        train_normal_only=False,
        target="next_day_log_return",
    )
    dataset = JointWindowDataset(
        manifest,
        window_store,
        scaler,
        window_size=4,
        normalize_batch=False,
        split="train",
        train_normal_only=False,
    )
    item = dataset[0]
    target_idx = item["meta"]["target_idx"]

    assert "target_return" in item
    assert target_idx == item["meta"]["start_idx"] + 4
    assert item["target_return"].item() == pytest.approx(window_store["AAA"]["returns"][target_idx])


def test_discover_tickers_and_window_labels(tmp_path: Path) -> None:
    make_ticker_files(tmp_path, ticker="BBB", rows=12)

    assert discover_tickers(str(tmp_path)) == ["BBB"]

    frame = pd.read_csv(tmp_path / "BBB_ohlcv.csv").merge(pd.read_csv(tmp_path / "BBB_anomaly_label.csv"), on="Date")
    _, _, _, _, _, records = build_windows_for_ticker(frame, "BBB", window_size=6, step=3, features="all")

    assert len(records) >= 2
    assert any(record.y_true == 1 for record in records)
