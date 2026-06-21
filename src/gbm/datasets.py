#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        def fit(self, values):
            array = np.asarray(values, dtype=float)
            self.mean_ = array.mean(axis=0)
            self.scale_ = array.std(axis=0)
            self.scale_ = np.where(self.scale_ <= 1e-12, 1.0, self.scale_)
            return self

        def transform(self, values):
            return (np.asarray(values, dtype=float) - self.mean_) / self.scale_
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None

    class Dataset:
        pass

from utils.validation_helpers import detect_label_columns

FEATURE_COLUMNS = {
    "all": ["Open", "High", "Low", "Close", "Volume", "LogReturn", "LogVolume", "RealizedVol"],
    "close_only": ["Close"],
    "log_return_only": ["LogReturn"],
    "log_return_volume": ["log_return_z", "volume_z"],
    "log_return_volume_vol": ["log_return_z", "volume_z", "rolling_volatility_20_z"],
    "log_return_tail_vol": [
        "log_return_z",
        "abs_log_return_z",
        "squared_log_return_z",
        "rolling_volatility_5_z",
        "rolling_volatility_20_z",
        "rolling_volatility_60_z",
        "return_z_20_z",
        "volume_z",
    ],
    "price_only": ["Open", "High", "Low", "Close", "LogReturn", "RealizedVol"],
    "volume_only": ["Volume", "LogReturn", "LogVolume", "RealizedVol"],
}


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    close = enriched["Close"].astype(float)
    volume = enriched["Volume"].astype(float)
    log_close = np.log(close.replace(0, np.nan))
    log_return = log_close.diff().fillna(0.0)
    log_volume = np.log1p(volume).fillna(0.0)
    realized_vol = log_return.rolling(5, min_periods=1).std(ddof=0).shift(1).fillna(0.0)
    return_mean_20 = log_return.rolling(20, min_periods=1).mean().shift(1)
    return_std_20 = log_return.rolling(20, min_periods=1).std(ddof=0).shift(1).replace(0, np.nan)
    enriched["LogReturn"] = log_return
    enriched["LogVolume"] = log_volume
    enriched["RealizedVol"] = realized_vol
    enriched["log_return_z"] = log_return
    enriched["abs_log_return_z"] = log_return.abs()
    enriched["squared_log_return_z"] = log_return**2
    enriched["rolling_volatility_5_z"] = log_return.rolling(5, min_periods=1).std(ddof=0).shift(1).fillna(0.0)
    enriched["rolling_volatility_20_z"] = log_return.rolling(20, min_periods=1).std(ddof=0).shift(1).fillna(0.0)
    enriched["rolling_volatility_60_z"] = log_return.rolling(60, min_periods=1).std(ddof=0).shift(1).fillna(0.0)
    enriched["return_z_20_z"] = ((log_return - return_mean_20) / return_std_20).fillna(0.0)
    volume_mean = volume.rolling(20, min_periods=1).mean().shift(1)
    volume_std = volume.rolling(20, min_periods=1).std(ddof=0).shift(1).replace(0, np.nan)
    enriched["volume_z"] = ((volume - volume_mean) / volume_std).fillna(0.0)
    return enriched


def get_feature_columns(features: str) -> List[str]:
    if features not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature set: {features}")
    return FEATURE_COLUMNS[features]


def get_output_root(output_root: Optional[str | Path] = None) -> Path:
    root = output_root or os.environ.get("AT_OUTPUT_ROOT") or "results"
    return Path(root).expanduser()


def get_run_dir(exp_name: str, output_root: Optional[str | Path] = None) -> Path:
    return get_output_root(output_root) / "experiments" / exp_name


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_std(series: pd.Series) -> pd.Series:
    return series.replace(0, np.nan)


def prepare_daily_feature_frame(
    frame: pd.DataFrame,
    volume_window: int = 20,
    volatility_window: int = 20,
    anomaly_baseline_window: int = 60,
    z_threshold: float = 3.0,
    high_swing_abs_log_return: float = 0.1,
) -> pd.DataFrame:
    """Build causal daily features and rule labels from OHLCV only."""
    out = frame.copy()
    if out.columns[0] != "Date":
        out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    required = {"Close", "Volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")
    out = out.dropna(subset=["Date", "Close", "Volume"]).reset_index(drop=True)
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))

    volume_mean = out["Volume"].rolling(volume_window, min_periods=volume_window).mean().shift(1)
    volume_std = out["Volume"].rolling(volume_window, min_periods=volume_window).std(ddof=0).shift(1)
    out["volume_z"] = (out["Volume"] - volume_mean) / _safe_std(volume_std)

    out["abs_log_return"] = out["log_return"].abs()
    out["squared_log_return"] = out["log_return"] ** 2
    out["rolling_volatility_5"] = out["log_return"].rolling(5, min_periods=5).std(ddof=0).shift(1)
    out["rolling_volatility_20"] = out["log_return"].rolling(volatility_window, min_periods=volatility_window).std(ddof=0).shift(1)
    out["rolling_volatility_60"] = out["log_return"].rolling(60, min_periods=60).std(ddof=0).shift(1)
    return_mean_20 = out["log_return"].rolling(20, min_periods=20).mean().shift(1)
    return_std_20 = out["log_return"].rolling(20, min_periods=20).std(ddof=0).shift(1)
    out["return_z_20"] = (out["log_return"] - return_mean_20) / _safe_std(return_std_20)
    vol_mean = out["rolling_volatility_20"].rolling(anomaly_baseline_window, min_periods=anomaly_baseline_window).mean().shift(1)
    vol_std = out["rolling_volatility_20"].rolling(anomaly_baseline_window, min_periods=anomaly_baseline_window).std(ddof=0).shift(1)
    out["rolling_volatility_20_rolling_z"] = (out["rolling_volatility_20"] - vol_mean) / _safe_std(vol_std)

    feature_cols = ["log_return", "abs_log_return", "squared_log_return", "volume_z", "rolling_volatility_5", "rolling_volatility_20", "rolling_volatility_60", "return_z_20"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).reset_index(drop=True)
    for column in feature_cols:
        mean = float(out[column].mean())
        std = float(out[column].std(ddof=0))
        out[f"{column}_z"] = 0.0 if std <= 1e-12 else (out[column] - mean) / std

    out["log_return_anomaly"] = 0
    out["volume_anomaly"] = (out["volume_z"].abs() >= z_threshold).fillna(False).astype(int)
    out["high_swing"] = (out["log_return"].abs() > high_swing_abs_log_return).fillna(False).astype(int)
    out["is_anomaly"] = (out[["log_return_anomaly", "volume_anomaly", "high_swing"]].sum(axis=1) > 0).astype(int)
    return out


def _window_split_spans(n_rows: int, window_size: int, step: int) -> dict[str, tuple[int, int] | None]:
    starts = list(range(0, max(1, n_rows - window_size + 1), step))
    if n_rows < window_size:
        starts = [0]
    effective_gap = math.ceil((window_size - 1) / max(1, step))
    n_total = len(starts)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    train_end = n_train
    val_start = min(n_total, train_end + effective_gap)
    val_end = min(n_total, val_start + n_val)
    test_start = min(n_total, val_end + effective_gap)

    def span(start_values: list[int]) -> tuple[int, int] | None:
        if not start_values:
            return None
        return start_values[0], min(start_values[-1] + window_size - 1, n_rows - 1)

    return {
        "train": span(starts[:train_end]),
        "val": span(starts[val_start:val_end]),
        "test": span(starts[test_start:]),
    }


def add_split_aware_log_return_labels(
    frame: pd.DataFrame,
    window_size: int,
    step: int,
    z_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = frame.copy()
    out["split"] = "unused"
    out["log_return_anomaly"] = 0
    stats_rows: list[dict[str, object]] = []
    spans = _window_split_spans(len(out), window_size=window_size, step=step)
    for split, span in spans.items():
        if span is None:
            continue
        start, end = span
        mask = (out.index >= start) & (out.index <= end)
        out.loc[mask, "split"] = split
        values = pd.to_numeric(out.loc[mask, "log_return"], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            upper = float("nan")
            lower = float("nan")
            expected = pd.Series(False, index=values.index)
        else:
            upper = mean + z_threshold * std
            lower = mean - z_threshold * std
            expected = (values > upper) | (values < lower)
        out.loc[mask, "log_return_anomaly"] = expected.fillna(False).astype(int)
        stats_rows.append(
            {
                "split": split,
                "start_idx": int(start),
                "end_idx": int(end),
                "rows": int(mask.sum()),
                "log_return_mean": mean,
                "log_return_std": std,
                "upper_3std": upper,
                "lower_3std": lower,
                "log_return_anomaly_days": int(expected.fillna(False).sum()),
            }
        )
    out["is_anomaly"] = (out[["log_return_anomaly", "volume_anomaly", "high_swing"]].sum(axis=1) > 0).astype(int)
    return out, stats_rows


def prepare_stock_feature_dataset(args) -> dict[str, object]:
    source_dir = Path(getattr(args, "source_data_path", "datasets/SP500"))
    output_dir = Path(getattr(args, "prepared_data_path", f"datasets/SP500_logreturn_volume_w{args.window_size}"))
    window_size = int(getattr(args, "window_size", 60))
    step = int(getattr(args, "step", 1))
    include_volatility_raw = getattr(args, "include_volatility", True)
    include_volatility = str(include_volatility_raw).lower() not in {"false", "0", "no", "off"}
    volume_window = int(getattr(args, "volume_window", 20))
    volatility_window = int(getattr(args, "volatility_window", 20))
    anomaly_baseline_window = int(getattr(args, "anomaly_baseline_window", 60))
    z_threshold = float(getattr(args, "label_z_threshold", 3.0))
    high_swing_abs_log_return = float(getattr(args, "high_swing_abs_log_return", 0.1))

    tickers = list(getattr(args, "tickers", None) or [])
    if not tickers:
        tickers = sorted(path.stem.replace("_ohlcv", "") for path in source_dir.glob("*_ohlcv.csv"))
    if not tickers:
        raise ValueError(f"No OHLCV files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = output_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    split_stat_rows: list[dict[str, object]] = []
    feature_columns = ["log_return_z", "volume_z"]
    if include_volatility:
        feature_columns.extend([
            "abs_log_return_z",
            "squared_log_return_z",
            "rolling_volatility_5_z",
            "rolling_volatility_20_z",
            "rolling_volatility_60_z",
            "return_z_20_z",
        ])

    sample_id = 0
    for ticker in tickers:
        source_path = source_dir / f"{ticker}_ohlcv.csv"
        if not source_path.exists():
            continue
        raw = pd.read_csv(source_path)
        prepared = prepare_daily_feature_frame(
            raw,
            volume_window=volume_window,
            volatility_window=volatility_window,
            anomaly_baseline_window=anomaly_baseline_window,
            z_threshold=z_threshold,
            high_swing_abs_log_return=high_swing_abs_log_return,
        )
        if len(prepared) < window_size:
            continue
        prepared, ticker_split_stats = add_split_aware_log_return_labels(
            prepared,
            window_size=window_size,
            step=step,
            z_threshold=z_threshold,
        )
        for stats_row in ticker_split_stats:
            split_stat_rows.append({"ticker": ticker, **stats_row})

        ohlcv_columns = [
            col for col in ["Date", "Open", "High", "Low", "Close", "Volume"] if col in prepared.columns
        ]
        feature_output_columns = [
            "log_return",
            "volume_z",
            "rolling_volatility_20",
            "return_z_20",
            "log_return_z",
            "rolling_volatility_20_z",
            "return_z_20_z",
            "rolling_volatility_20_rolling_z",
        ]
        prepared[ohlcv_columns + feature_output_columns].to_csv(output_dir / f"{ticker}_ohlcv.csv", index=False)
        labels = prepared[["Date", "split", "log_return_anomaly", "volume_anomaly", "high_swing"]].copy()
        labels.to_csv(output_dir / f"{ticker}_anomaly_label.csv", index=False)

        x_values = prepared[feature_columns].astype(float).to_numpy()
        y_values = prepared["is_anomaly"].astype(int).to_numpy()
        dates = prepared["Date"].astype(str).to_numpy()
        starts = list(range(0, len(prepared) - window_size + 1, step))
        windows = np.stack([x_values[start: start + window_size] for start in starts], axis=0)
        y_window = np.asarray([int(y_values[start: start + window_size].max()) for start in starts], dtype=np.int64)
        np.savez_compressed(
            windows_dir / f"{ticker}_windows_w{window_size}.npz",
            X=windows,
            y=y_window,
            dates=dates,
            start_idx=np.asarray(starts, dtype=np.int64),
            end_idx=np.asarray([start + window_size - 1 for start in starts], dtype=np.int64),
            feature_names=np.asarray(feature_columns),
        )
        for start, y_value in zip(starts, y_window):
            end = start + window_size - 1
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "ticker": ticker,
                    "window_size": window_size,
                    "start_idx": start,
                    "end_idx": end,
                    "start_date": dates[start],
                    "end_date": dates[end],
                    "y_true": int(y_value),
                }
            )
            sample_id += 1
        summaries.append(
            {
                "ticker": ticker,
                "rows": int(len(prepared)),
                "windows": int(len(starts)),
                "anomaly_days": int(prepared["is_anomaly"].sum()),
                "anomaly_windows": int(y_window.sum()),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summaries)
    split_stats = pd.DataFrame(split_stat_rows)
    manifest.to_csv(output_dir / f"window_manifest_w{window_size}.csv", index=False)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    split_stats.to_csv(output_dir / "split_log_return_thresholds.csv", index=False)
    save_json(
        output_dir / "dataset_metadata.json",
        {
            "source_data_path": str(source_dir),
            "window_size": window_size,
            "step": step,
            "feature_columns": feature_columns,
            "volume_window": volume_window,
            "volatility_window": volatility_window,
            "anomaly_baseline_window": anomaly_baseline_window,
            "label_z_threshold": z_threshold,
            "high_swing_abs_log_return": high_swing_abs_log_return,
            "tickers": int(len(summary)),
            "windows": int(len(manifest)),
            "label_rule": "diagnostic labels: log_return_anomaly = raw log_return outside mean(split) +/- 3*std(split) for that ticker and split; volume_anomaly = abs(causal volume_z) >= threshold; high_swing = abs(log_return) > high_swing_abs_log_return",
        },
    )
    print(f"Prepared {len(summary)} tickers, windows={len(manifest)} at {output_dir}")
    return {"output_dir": str(output_dir), "tickers": int(len(summary)), "windows": int(len(manifest))}


@dataclass
class JointWindowRecord:
    sample_id: int
    ticker: str
    split: str
    start_idx: int
    end_idx: int
    target_idx: int
    start_date: str
    end_date: str
    target_date: str
    y_true: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def discover_tickers(data_path: str) -> List[str]:
    tickers = []
    root = Path(data_path)
    for path in sorted(root.glob("*_ohlcv.csv")):
        ticker = path.stem.replace("_ohlcv", "")
        if (root / f"{ticker}_anomaly_label.csv").exists():
            tickers.append(ticker)
    return tickers


def load_ticker_frame(data_path: str, ticker: str) -> pd.DataFrame:
    ohlcv_path = Path(data_path) / f"{ticker}_ohlcv.csv"
    label_path = Path(data_path) / f"{ticker}_anomaly_label.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing OHLCV file: {ohlcv_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    ohlcv = pd.read_csv(ohlcv_path)
    labels = pd.read_csv(label_path)
    if ohlcv.columns[0] != "Date":
        ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "Date"})
    if labels.columns[0] != "Date":
        labels = labels.rename(columns={labels.columns[0]: "Date"})
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    labels["Date"] = pd.to_datetime(labels["Date"])
    frame = ohlcv.merge(labels, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    return frame


def get_label_vector(frame: pd.DataFrame) -> np.ndarray:
    label_columns = [col for col in detect_label_columns(frame) if col != "Date"]
    if not label_columns:
        return np.zeros(len(frame), dtype=np.int64)
    return (frame[label_columns].fillna(0).to_numpy() > 0).any(axis=1).astype(np.int64)


def build_windows_for_ticker(
    frame: pd.DataFrame,
    ticker: str,
    window_size: int,
    step: int,
    features: str,
    target: str = "window_returns",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[JointWindowRecord]]:
    if target not in {"window_returns", "next_day_log_return"}:
        raise ValueError("target must be window_returns or next_day_log_return")
    enriched = add_derived_features(frame)
    feature_columns = get_feature_columns(features)
    x = np.nan_to_num(enriched[feature_columns].astype(float).to_numpy())
    returns = enriched["LogReturn"].astype(float).to_numpy()
    date_series = pd.to_datetime(enriched["Date"])
    raw_day_deltas = date_series.diff().dt.days.astype(float).fillna(np.nan).to_numpy()
    positive_deltas = raw_day_deltas[np.isfinite(raw_day_deltas) & (raw_day_deltas > 0)]
    base_delta = float(np.median(positive_deltas)) if len(positive_deltas) else 1.0
    time_deltas = np.nan_to_num(raw_day_deltas / max(base_delta, 1e-8), nan=1.0, posinf=1.0, neginf=1.0)
    time_deltas = np.clip(time_deltas, 1e-6, None)
    dates = enriched["Date"].astype(str).tolist()
    labels = get_label_vector(frame)

    max_start = len(enriched) - window_size if target == "next_day_log_return" else len(enriched) - window_size + 1
    indices = list(range(0, max(1, max_start), step))
    if len(enriched) < window_size:
        indices = [0]

    records: List[JointWindowRecord] = []
    for window_id, start in enumerate(indices):
        end = start + window_size
        start_idx = int(start)
        end_idx = int(min(end, len(enriched)))
        target_idx = int(min(end, len(enriched) - 1)) if target == "next_day_log_return" else int(min(end - 1, len(enriched) - 1))
        y_window = labels[target_idx: target_idx + 1] if target == "next_day_log_return" else labels[start:end]
        if target == "window_returns" and len(y_window) < window_size:
            y_window = np.pad(y_window, (0, window_size - len(y_window)), mode="constant")
        records.append(
            JointWindowRecord(
                sample_id=-1,
                ticker=ticker,
                split="",
                start_idx=start_idx,
                end_idx=end_idx,
                target_idx=target_idx,
                start_date=dates[start_idx] if start_idx < len(dates) else dates[-1],
                end_date=dates[target_idx] if target == "next_day_log_return" and dates else dates[min(end - 1, len(dates) - 1)] if dates else "",
                target_date=dates[target_idx] if dates else "",
                y_true=int(np.any(y_window > 0)),
            )
        )
    return enriched, x, returns, time_deltas, labels, records


def create_joint_manifest(
    data_path: str,
    tickers: Optional[Sequence[str]],
    window_size: int,
    step: int,
    features: str,
    seed: int,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_method: str = "chronological",
    purge_gap: Optional[int] = None,
    train_normal_only: bool = True,
    target: str = "window_returns",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], StandardScaler]:
    if split_method not in {"chronological", "random"}:
        raise ValueError("split_method must be chronological or random")
    selected_tickers = list(tickers) if tickers else discover_tickers(data_path)
    if not selected_tickers:
        raise SystemExit(f"No tickers found in {data_path}")

    window_store: Dict[str, Dict[str, np.ndarray]] = {}
    manifest_rows: List[Dict[str, object]] = []
    sample_rows: List[np.ndarray] = []
    sample_index = 0

    for ticker in selected_tickers:
        frame = load_ticker_frame(data_path, ticker)
        enriched, x, returns, time_deltas, labels, records = build_windows_for_ticker(
            frame,
            ticker,
            window_size,
            step,
            features,
            target=target,
        )
        window_store[ticker] = {
            "x": x,
            "returns": returns,
            "time_deltas": time_deltas,
            "labels": labels,
            "dates": enriched["Date"].astype(str).to_numpy(),
        }
        for record in records:
            record.sample_id = sample_index
            sample_index += 1
            sample_rows.append(x[record.start_idx: record.start_idx + window_size])
            manifest_rows.append(
                {
                    "sample_id": record.sample_id,
                    "target": target,
                    "ticker": record.ticker,
                    "start_idx": record.start_idx,
                    "end_idx": record.end_idx,
                    "target_idx": record.target_idx,
                    "start_date": record.start_date,
                    "end_date": record.end_date,
                    "target_date": record.target_date,
                    "y_true": record.y_true,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    if split_method == "random":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(manifest))
        n_total = len(manifest)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        train_ids = set(permutation[:n_train].tolist())
        val_ids = set(permutation[n_train: n_train + n_val].tolist())
        split_values = []
        for sample_id in manifest["sample_id"].tolist():
            if sample_id in train_ids:
                split_values.append("train")
            elif sample_id in val_ids:
                split_values.append("val")
            else:
                split_values.append("test")
        manifest["split"] = split_values
    else:
        effective_gap = math.ceil((window_size - 1) / max(1, step)) if purge_gap is None else max(0, purge_gap)
        manifest["split"] = "purged"
        for _, group in manifest.groupby("ticker", sort=False):
            ordered = group.sort_values("start_idx")
            n_total = len(ordered)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])
            train_end = n_train
            val_start = min(n_total, train_end + effective_gap)
            val_end = min(n_total, val_start + n_val)
            test_start = min(n_total, val_end + effective_gap)
            manifest.loc[ordered.iloc[:train_end].index, "split"] = "train"
            manifest.loc[ordered.iloc[val_start:val_end].index, "split"] = "val"
            manifest.loc[ordered.iloc[test_start:].index, "split"] = "test"
        manifest = manifest[manifest["split"].isin({"train", "val", "test"})].reset_index(drop=True)

    train_mask = manifest["split"] == "train"
    if train_normal_only:
        train_mask = train_mask & (manifest["y_true"] == 0)
    train_windows = [sample_rows[int(sample_id)] for sample_id in manifest.loc[train_mask, "sample_id"].tolist()]
    if not train_windows:
        raise RuntimeError("No train windows were created")
    train_stack = np.concatenate(train_windows, axis=0)
    scaler = StandardScaler().fit(train_stack)

    return manifest, window_store, scaler


class JointWindowDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        window_store: Dict[str, Dict[str, np.ndarray]],
        scaler: StandardScaler,
        window_size: int,
        normalize_batch: bool,
        split: str,
        train_normal_only: bool = True,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        split_manifest = manifest[manifest["split"] == split]
        if split == "train" and train_normal_only:
            split_manifest = split_manifest[split_manifest["y_true"] == 0]
        self.manifest = split_manifest.reset_index(drop=True)
        self.window_store = window_store
        self.scaler = scaler
        self.window_size = window_size
        self.normalize_batch = normalize_batch
        self.split = split

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        ticker = row["ticker"]
        ticker_store = self.window_store[ticker]
        start = int(row["start_idx"])
        end = start + self.window_size
        target_idx = int(row.get("target_idx", min(end - 1, len(ticker_store["returns"]) - 1)))

        x_window = ticker_store["x"][start:end]
        returns_window = ticker_store["returns"][start:end]
        target_return = float(ticker_store["returns"][target_idx])
        time_deltas_window = ticker_store["time_deltas"][start:end]
        labels_window = ticker_store["labels"][start:end]

        if len(x_window) < self.window_size:
            pad_len = self.window_size - len(x_window)
            x_window = np.pad(x_window, ((0, pad_len), (0, 0)), mode="edge")
            returns_window = np.pad(returns_window, (0, pad_len), mode="edge" if len(returns_window) else "constant")
            time_deltas_window = np.pad(time_deltas_window, (0, pad_len), mode="edge" if len(time_deltas_window) else "constant")
            labels_window = np.pad(labels_window, (0, pad_len), mode="constant")

        x_window = self.scaler.transform(x_window)
        if self.normalize_batch:
            mean = x_window.mean(axis=0, keepdims=True)
            std = x_window.std(axis=0, keepdims=True) + 1e-8
            x_window = (x_window - mean) / std

        meta = {
            "ticker": ticker,
            "split": self.split,
            "window_id": int(row["sample_id"]),
            "start_idx": int(row["start_idx"]),
            "end_idx": int(row["end_idx"]),
            "target_idx": target_idx,
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "target_date": row.get("target_date", row["end_date"]),
        }
        item = {
            "x": torch.tensor(x_window, dtype=torch.float32),
            "returns": torch.tensor(returns_window, dtype=torch.float32),
            "time_deltas": torch.tensor(time_deltas_window, dtype=torch.float32),
            "y": torch.tensor(int(row["y_true"]), dtype=torch.float32),
            "meta": meta,
        }
        if str(row.get("target", "window_returns")) == "next_day_log_return":
            item["target_return"] = torch.tensor(target_return, dtype=torch.float32)
        return item


def build_joint_loaders(
    data_path: str,
    tickers: Optional[Sequence[str]],
    window_size: int,
    batch_size: int,
    step: int,
    features: str,
    normalize_batch: bool,
    seed: int,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    split_method: str = "chronological",
    purge_gap: Optional[int] = None,
    train_normal_only: bool = True,
    target: str = "window_returns",
):
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to build joint loaders")
    manifest, window_store, scaler = create_joint_manifest(
        data_path=data_path,
        tickers=tickers,
        window_size=window_size,
        step=step,
        features=features,
        seed=seed,
        split_ratios=split_ratios,
        split_method=split_method,
        purge_gap=purge_gap,
        train_normal_only=train_normal_only,
        target=target,
    )

    train_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "train", train_normal_only=train_normal_only)
    val_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "val", train_normal_only=False)
    test_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "test", train_normal_only=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    input_dim = window_store[next(iter(window_store))]["x"].shape[1]
    return manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim


def save_joint_manifest(
    run_dir: Path,
    manifest: pd.DataFrame,
    seed: int,
    window_size: int,
    step: int,
    features: str,
    split_method: str = "chronological",
    purge_gap: Optional[int] = None,
    train_normal_only: bool = True,
    target: str = "window_returns",
) -> Path:
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"joint_split_seed{seed}_w{window_size}_s{step}_{features}.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = manifest.groupby(["split", "ticker"]).size().reset_index(name="windows")
    save_json(
        split_dir / f"joint_split_seed{seed}_w{window_size}_s{step}_{features}.json",
        {
            "seed": seed,
            "window_size": window_size,
            "step": step,
            "features": features,
            "split_method": split_method,
            "purge_gap": purge_gap,
            "train_normal_only": train_normal_only,
            "target": target,
            "total_windows": int(len(manifest)),
            "split_counts": manifest["split"].value_counts().to_dict(),
            "ticker_split_counts": summary.to_dict(orient="records"),
        },
    )
    return manifest_path

__all__ = [
    "FEATURE_COLUMNS",
    "JointWindowDataset",
    "JointWindowRecord",
    "add_derived_features",
    "build_joint_loaders",
    "build_windows_for_ticker",
    "create_joint_manifest",
    "discover_tickers",
    "get_feature_columns",
    "get_label_vector",
    "get_output_root",
    "get_run_dir",
    "load_ticker_frame",
    "prepare_daily_feature_frame",
    "prepare_stock_feature_dataset",
    "save_joint_manifest",
    "save_json",
    "set_seed",
]
