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
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.validation_helpers import detect_label_columns

FEATURE_COLUMNS = {
    "all": ["Open", "High", "Low", "Close", "Volume", "LogReturn", "LogVolume", "RealizedVol"],
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
    realized_vol = log_return.rolling(5, min_periods=1).std(ddof=0).fillna(0.0)
    enriched["LogReturn"] = log_return
    enriched["LogVolume"] = log_volume
    enriched["RealizedVol"] = realized_vol
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


@dataclass
class JointWindowRecord:
    sample_id: int
    ticker: str
    split: str
    start_idx: int
    end_idx: int
    start_date: str
    end_date: str
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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[JointWindowRecord]]:
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

    indices = list(range(0, max(1, len(enriched) - window_size + 1), step))
    if len(enriched) < window_size:
        indices = [0]

    records: List[JointWindowRecord] = []
    for window_id, start in enumerate(indices):
        end = start + window_size
        start_idx = int(start)
        end_idx = int(min(end, len(enriched)))
        y_window = labels[start:end]
        if len(y_window) < window_size:
            y_window = np.pad(y_window, (0, window_size - len(y_window)), mode="constant")
        records.append(
            JointWindowRecord(
                sample_id=-1,
                ticker=ticker,
                split="",
                start_idx=start_idx,
                end_idx=end_idx,
                start_date=dates[start_idx] if start_idx < len(dates) else dates[-1],
                end_date=dates[min(end - 1, len(dates) - 1)] if dates else "",
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
        enriched, x, returns, time_deltas, labels, records = build_windows_for_ticker(frame, ticker, window_size, step, features)
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
                    "ticker": record.ticker,
                    "start_idx": record.start_idx,
                    "end_idx": record.end_idx,
                    "start_date": record.start_date,
                    "end_date": record.end_date,
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

        x_window = ticker_store["x"][start:end]
        returns_window = ticker_store["returns"][start:end]
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
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        }
        return {
            "x": torch.tensor(x_window, dtype=torch.float32),
            "returns": torch.tensor(returns_window, dtype=torch.float32),
            "time_deltas": torch.tensor(time_deltas_window, dtype=torch.float32),
            "y": torch.tensor(int(row["y_true"]), dtype=torch.float32),
            "meta": meta,
        }


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
):
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
    "save_joint_manifest",
    "save_json",
    "set_seed",
]
