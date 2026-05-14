from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


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

