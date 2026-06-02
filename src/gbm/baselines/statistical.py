from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

EPS = 1e-8


def rolling_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=0))


def mean_abs_return(returns: np.ndarray) -> float:
    return float(np.mean(np.abs(returns)))


def last_return_zscore(returns: np.ndarray) -> float:
    if len(returns) <= 1:
        return 0.0
    history = returns[:-1]
    return float(abs((returns[-1] - np.mean(history)) / (np.std(history, ddof=0) + EPS)))


def max_return_zscore(returns: np.ndarray) -> float:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    return float(np.max(np.abs((returns - mean) / (std + EPS))))


def ewma_volatility(returns: np.ndarray, span: float = 20.0) -> float:
    if len(returns) == 0:
        return 0.0
    weights = np.exp(-np.arange(len(returns))[::-1] / max(span, 1.0))
    weights = weights / weights.sum()
    mean = float(np.sum(weights * returns))
    var = float(np.sum(weights * (returns - mean) ** 2))
    return math.sqrt(max(var, 0.0))


def vol_ratio_short_long(returns: np.ndarray, short: int = 5, long: int = 20) -> float:
    if len(returns) < long:
        return float(np.std(returns, ddof=0))
    short_vol = float(np.std(returns[-short:], ddof=0))
    long_vol = float(np.std(returns[-long:], ddof=0))
    return short_vol / (long_vol + EPS)


def cusum_abs_return(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    centered = returns - np.median(returns)
    cumulative = np.cumsum(centered)
    return float(np.max(cumulative) - np.min(cumulative))


STATISTICAL_BASELINES: Dict[str, Callable[[np.ndarray], float]] = {
    "rolling_volatility": rolling_volatility,
    "mean_abs_return": mean_abs_return,
    "last_return_zscore": last_return_zscore,
    "max_return_zscore": max_return_zscore,
    "ewma_volatility": ewma_volatility,
    "vol_ratio_short_long": vol_ratio_short_long,
    "cusum_abs_return": cusum_abs_return,
}

BASELINE_NAMES = sorted(STATISTICAL_BASELINES)


def score_manifest(
    manifest: "pd.DataFrame",
    window_store: dict,
    baseline_name: str,
    scorer: Callable[[np.ndarray], float],
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for row in manifest.itertuples(index=False):
        ticker_store = window_store[row.ticker]
        span = int(row.end_idx - row.start_idx)
        returns = ticker_store["returns"][int(row.start_idx) : int(row.start_idx) + span]
        score = scorer(np.nan_to_num(returns.astype(float)))
        if not math.isfinite(score):
            score = 0.0
        rows.append(
            {
                "baseline": baseline_name,
                "sample_id": int(row.sample_id),
                "ticker": row.ticker,
                "split": row.split,
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "start_date": row.start_date,
                "end_date": row.end_date,
                "y_true": int(row.y_true),
                "score": float(score),
            }
        )
    return pd.DataFrame(rows)
