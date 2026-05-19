#!/usr/bin/env python3
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.gbm.features import add_derived_features, get_feature_columns
from src.gbm.io import save_json
from src.gbm.paths import get_output_root, get_run_dir
from utils.validation_helpers import detect_label_columns


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
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[JointWindowRecord]]:
    enriched = add_derived_features(frame)
    feature_columns = get_feature_columns(features)
    x = np.nan_to_num(enriched[feature_columns].astype(float).to_numpy())
    returns = enriched["LogReturn"].astype(float).to_numpy()
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
    return enriched, x, returns, labels, records


def create_joint_manifest(
    data_path: str,
    tickers: Optional[Sequence[str]],
    window_size: int,
    step: int,
    features: str,
    seed: int,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], StandardScaler]:
    selected_tickers = list(tickers) if tickers else discover_tickers(data_path)
    if not selected_tickers:
        raise SystemExit(f"No tickers found in {data_path}")

    window_store: Dict[str, Dict[str, np.ndarray]] = {}
    manifest_rows: List[Dict[str, object]] = []
    sample_rows: List[np.ndarray] = []
    sample_index = 0

    for ticker in selected_tickers:
        frame = load_ticker_frame(data_path, ticker)
        enriched, x, returns, labels, records = build_windows_for_ticker(frame, ticker, window_size, step, features)
        window_store[ticker] = {
            "x": x,
            "returns": returns,
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
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(manifest))
    n_total = len(manifest)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    train_ids = set(permutation[:n_train].tolist())
    val_ids = set(permutation[n_train: n_train + n_val].tolist())
    test_ids = set(permutation[n_train + n_val :].tolist())

    split_values = []
    for sample_id in manifest["sample_id"].tolist():
        if sample_id in train_ids:
            split_values.append("train")
        elif sample_id in val_ids:
            split_values.append("val")
        else:
            split_values.append("test")
    manifest["split"] = split_values

    train_mask = manifest["split"] == "train"
    train_windows = [sample_rows[i] for i, keep in enumerate(train_mask.tolist()) if keep]
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
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
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
        labels_window = ticker_store["labels"][start:end]

        if len(x_window) < self.window_size:
            pad_len = self.window_size - len(x_window)
            x_window = np.pad(x_window, ((0, pad_len), (0, 0)), mode="edge")
            returns_window = np.pad(returns_window, (0, pad_len), mode="edge" if len(returns_window) else "constant")
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
):
    manifest, window_store, scaler = create_joint_manifest(
        data_path=data_path,
        tickers=tickers,
        window_size=window_size,
        step=step,
        features=features,
        seed=seed,
        split_ratios=split_ratios,
    )

    train_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "train")
    val_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "val")
    test_ds = JointWindowDataset(manifest, window_store, scaler, window_size, normalize_batch, "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    input_dim = window_store[next(iter(window_store))]["x"].shape[1]
    return manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim


def save_joint_manifest(run_dir: Path, manifest: pd.DataFrame, seed: int, window_size: int, step: int, features: str) -> Path:
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
            "total_windows": int(len(manifest)),
            "split_counts": manifest["split"].value_counts().to_dict(),
            "ticker_split_counts": summary.to_dict(orient="records"),
        },
    )
    return manifest_path
