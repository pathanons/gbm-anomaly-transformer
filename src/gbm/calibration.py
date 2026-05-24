from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvtFit:
    threshold: float
    tail_base_threshold: float
    tail_base_quantile: float
    target_quantile: float
    shape: float
    scale: float
    exceedance_count: int
    calibration_count: int
    method: str


def normal_calibration_subset(frame: pd.DataFrame, score_column: str = "score") -> tuple[pd.DataFrame, str]:
    normal = frame.loc[frame["y_true"] == 0].copy()
    if not normal.empty:
        return normal, "validation_normal_windows"
    return frame.copy(), "validation_all_windows"


def empirical_threshold(scores: np.ndarray, quantile: float) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Cannot fit threshold from empty calibration scores")
    return float(np.quantile(clean, quantile))


def conformal_p_value(score: float, calibration_scores: np.ndarray) -> float:
    calibration = np.asarray(calibration_scores, dtype=float)
    calibration = calibration[np.isfinite(calibration)]
    if calibration.size == 0:
        return float("nan")
    return (float((calibration >= float(score)).sum()) + 1.0) / (float(calibration.size) + 1.0)


def add_conformal_p_values(
    frame: pd.DataFrame,
    calibration_scores: np.ndarray,
    score_column: str = "score",
    output_column: str = "conformal_p_value",
) -> pd.DataFrame:
    out = frame.copy()
    calibration = np.asarray(calibration_scores, dtype=float)
    out[output_column] = out[score_column].apply(lambda score: conformal_p_value(float(score), calibration))
    return out


def calibration_scores_by_ticker(
    frame: pd.DataFrame,
    score_column: str = "score",
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for ticker, ticker_frame in frame.groupby("ticker"):
        values = ticker_frame[score_column].astype(float).to_numpy()
        values = values[np.isfinite(values)]
        if values.size:
            grouped[str(ticker)] = [float(value) for value in values.tolist()]
    return grouped


def add_per_ticker_conformal_p_values(
    frame: pd.DataFrame,
    calibration_by_ticker: Mapping[str, list[float]],
    global_calibration_scores: np.ndarray,
    score_column: str = "score",
    output_column: str = "conformal_p_value",
) -> pd.DataFrame:
    out = frame.copy()
    global_scores = np.asarray(global_calibration_scores, dtype=float)

    def row_p_value(row: pd.Series) -> float:
        ticker_scores = calibration_by_ticker.get(str(row["ticker"]))
        calibration = np.asarray(ticker_scores, dtype=float) if ticker_scores else global_scores
        return conformal_p_value(float(row[score_column]), calibration)

    out[output_column] = out.apply(row_p_value, axis=1)
    return out


def fit_evt_threshold(
    scores: np.ndarray,
    target_quantile: float,
    tail_base_quantile: float = 0.90,
) -> EvtFit:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("Cannot fit EVT threshold from empty calibration scores")
    if not 0.0 < tail_base_quantile < 1.0:
        raise ValueError("tail_base_quantile must be between 0 and 1")
    if not 0.0 < target_quantile < 1.0:
        raise ValueError("target_quantile must be between 0 and 1")
    if target_quantile <= tail_base_quantile:
        threshold = empirical_threshold(clean, target_quantile)
        return EvtFit(
            threshold=threshold,
            tail_base_threshold=threshold,
            tail_base_quantile=tail_base_quantile,
            target_quantile=target_quantile,
            shape=float("nan"),
            scale=float("nan"),
            exceedance_count=0,
            calibration_count=int(clean.size),
            method="empirical_target_below_evt_tail",
        )

    base_threshold = empirical_threshold(clean, tail_base_quantile)
    exceedances = clean[clean > base_threshold] - base_threshold
    exceedances = exceedances[exceedances > 0]
    if exceedances.size < 10:
        threshold = empirical_threshold(clean, target_quantile)
        return EvtFit(
            threshold=threshold,
            tail_base_threshold=base_threshold,
            tail_base_quantile=tail_base_quantile,
            target_quantile=target_quantile,
            shape=float("nan"),
            scale=float("nan"),
            exceedance_count=int(exceedances.size),
            calibration_count=int(clean.size),
            method="empirical_too_few_exceedances",
        )

    try:
        from scipy.stats import genpareto

        shape, _, scale = genpareto.fit(exceedances, floc=0.0)
        exceedance_prob = float(exceedances.size) / float(clean.size)
        conditional_cdf = 1.0 - ((1.0 - target_quantile) / exceedance_prob)
        conditional_cdf = float(np.clip(conditional_cdf, 0.0, 1.0 - 1e-12))
        threshold = float(base_threshold + genpareto.ppf(conditional_cdf, shape, loc=0.0, scale=scale))
        if not np.isfinite(threshold):
            raise ValueError("Non-finite EVT threshold")
        return EvtFit(
            threshold=threshold,
            tail_base_threshold=base_threshold,
            tail_base_quantile=tail_base_quantile,
            target_quantile=target_quantile,
            shape=float(shape),
            scale=float(scale),
            exceedance_count=int(exceedances.size),
            calibration_count=int(clean.size),
            method="gpd_mle",
        )
    except Exception:
        threshold = empirical_threshold(clean, target_quantile)
        return EvtFit(
            threshold=threshold,
            tail_base_threshold=base_threshold,
            tail_base_quantile=tail_base_quantile,
            target_quantile=target_quantile,
            shape=float("nan"),
            scale=float("nan"),
            exceedance_count=int(exceedances.size),
            calibration_count=int(clean.size),
            method="empirical_evt_fit_failed",
        )


def evt_fit_to_json(fit: EvtFit) -> dict[str, float | int | str]:
    return {
        "threshold": fit.threshold,
        "tail_base_threshold": fit.tail_base_threshold,
        "tail_base_quantile": fit.tail_base_quantile,
        "target_quantile": fit.target_quantile,
        "shape": fit.shape,
        "scale": fit.scale,
        "exceedance_count": fit.exceedance_count,
        "calibration_count": fit.calibration_count,
        "method": fit.method,
    }

