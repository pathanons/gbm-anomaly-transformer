"""Explicit event-label helpers for the SP500 taxonomy dataset.

Use this module when a script needs to generate or document labels for:
- jump
- drop
- volume_spike
- volatility_shock
- regime_shift
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    lookback_window: int = 100
    short_window: int = 20
    regime_persistence: int = 5
    return_z: float = 2.5
    volume_z: float = 2.5
    volatility_z: float = 2.0
    regime_mean_z: float = 1.5
    regime_vol_ratio: float = 1.5


def build_event_labels(ohlcv: pd.DataFrame, config: LabelConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frame = ohlcv.copy()
    if "Date" not in frame.columns:
        raise ValueError("OHLCV file must contain a Date column")

    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    ret = log_close.diff()
    log_volume = np.log1p(volume)

    W = config.lookback_window
    short_w = config.short_window
    persistence = config.regime_persistence
    eps = 1e-8

    ret_mean = ret.rolling(W, min_periods=W).mean().shift(1)
    ret_std = ret.rolling(W, min_periods=W).std(ddof=0).shift(1)
    vol_mean = log_volume.rolling(W, min_periods=W).mean().shift(1)
    vol_std = log_volume.rolling(W, min_periods=W).std(ddof=0).shift(1)

    realized_vol = ret.rolling(short_w, min_periods=short_w).std(ddof=0)
    rv_mean = realized_vol.rolling(W, min_periods=W).mean().shift(1)
    rv_std = realized_vol.rolling(W, min_periods=W).std(ddof=0).shift(1)

    jump = ((ret - ret_mean) / (ret_std + eps) >= config.return_z).fillna(False)
    drop = ((ret - ret_mean) / (ret_std + eps) <= -config.return_z).fillna(False)
    volume_spike = ((log_volume - vol_mean) / (vol_std + eps) >= config.volume_z).fillna(False)
    volatility_shock = ((realized_vol - rv_mean) / (rv_std + eps) >= config.volatility_z).fillna(False)

    short_mean_ret = ret.rolling(short_w, min_periods=short_w).mean()
    long_mean_ret = ret.rolling(W, min_periods=W).mean().shift(1)
    long_std_ret = ret.rolling(W, min_periods=W).std(ddof=0).shift(1)
    long_rv = realized_vol.rolling(W, min_periods=W).mean().shift(1)

    mean_shift_score = (short_mean_ret - long_mean_ret).abs() / (long_std_ret + eps)
    vol_ratio = realized_vol / (long_rv + eps)
    regime_raw = (mean_shift_score >= config.regime_mean_z) & (vol_ratio >= config.regime_vol_ratio)
    regime_shift = regime_raw.rolling(persistence, min_periods=persistence).sum() >= persistence
    regime_shift = regime_shift.fillna(False)

    events = pd.DataFrame(
        {
            "Date": frame["Date"],
            "jump": jump.astype(int),
            "drop": drop.astype(int),
            "volume_spike": volume_spike.astype(int),
            "volatility_shock": volatility_shock.astype(int),
            "regime_shift": regime_shift.astype(int),
        }
    )
    events["is_anomaly"] = (events[["jump", "drop", "volume_spike", "volatility_shock", "regime_shift"]].sum(axis=1) > 0).astype(int)

    stats = {
        "rows": int(len(events)),
        "jump_count": int(events["jump"].sum()),
        "drop_count": int(events["drop"].sum()),
        "volume_spike_count": int(events["volume_spike"].sum()),
        "volatility_shock_count": int(events["volatility_shock"].sum()),
        "regime_shift_count": int(events["regime_shift"].sum()),
        "event_count": int(events["is_anomaly"].sum()),
        "event_rate": float(events["is_anomaly"].mean()),
    }
    return events, stats


def render_spec_markdown(config: LabelConfig) -> str:
    return f"""# Event Label Specification

This dataset was generated with a lookback window of {config.lookback_window} trading days.

## Parameters
- lookback_window = {config.lookback_window}
- short_window = {config.short_window}
- regime_persistence = {config.regime_persistence}
- return_z = {config.return_z}
- volume_z = {config.volume_z}
- volatility_z = {config.volatility_z}
- regime_mean_z = {config.regime_mean_z}
- regime_vol_ratio = {config.regime_vol_ratio}

## Label Definitions
- jump: positive return shock relative to the trailing {config.lookback_window}-day baseline
- drop: negative return shock relative to the trailing {config.lookback_window}-day baseline
- volume_spike: abnormal log-volume surge relative to the trailing {config.lookback_window}-day baseline
- volatility_shock: realized-volatility surge relative to the trailing {config.lookback_window}-day baseline
- regime_shift: persistent deviation in short-window return mean and realized volatility for at least {config.regime_persistence} days

## Notes
- All baselines use only historical data up to t-1.
- The composite anomaly label is the OR of all event columns.
- The dataset keeps the event labels at point level, so each day can have multiple concurrent events.
"""
