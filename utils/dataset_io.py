"""Dataset path and I/O helpers.

Use this module when a script needs to:
- discover SP500 tickers from a dataset directory
- resolve OHLCV and label file paths
- copy or write dataset artifacts in a consistent way
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def discover_sp500_tickers(data_path: Path) -> List[str]:
    return sorted(path.stem.replace("_ohlcv", "") for path in data_path.glob("*_ohlcv.csv"))


def ohlcv_path(data_path: Path, ticker: str) -> Path:
    return data_path / f"{ticker}_ohlcv.csv"


def anomaly_label_path(data_path: Path, ticker: str) -> Path:
    return data_path / f"{ticker}_anomaly_label.csv"


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
