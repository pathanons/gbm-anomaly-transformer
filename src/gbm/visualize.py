"""Canonical visualization surface for GBM experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.validation_helpers import detect_label_columns

def contiguous_intervals(dates, mask) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start_idx = None
    date_list = list(pd.to_datetime(dates))
    mask_list = list(mask)

    for idx, flagged in enumerate(mask_list):
        if flagged and start_idx is None:
            start_idx = idx
        elif not flagged and start_idx is not None:
            intervals.append((date_list[start_idx], date_list[idx - 1]))
            start_idx = None

    if start_idx is not None:
        intervals.append((date_list[start_idx], date_list[len(mask_list) - 1]))
    return intervals


def interval_end(dates, end_date) -> pd.Timestamp:
    date_list = list(pd.to_datetime(dates))
    try:
        end_idx = date_list.index(pd.to_datetime(end_date))
    except ValueError:
        return pd.to_datetime(end_date)
    if end_idx + 1 < len(date_list):
        return date_list[end_idx + 1]
    return date_list[end_idx]


MAD_SCALE = 1.4826


@dataclass(frozen=True)
class MadThreshold:
    threshold: float
    median: float
    mad: float


@dataclass(frozen=True)
class SpikeSummary:
    ticker: str
    model_spikes: int
    price_anoms: int
    overlaps: int
    threshold: float
    score_median: float
    score_mad: float
    threshold_mode: str
    outpath: str


@dataclass(frozen=True)
class AttentionPlotSummary:
    ticker: str
    window_id: int
    layer: int
    head: int
    endpoint_top_lag: int
    endpoint_top_weight: float
    endpoint_l1_mean: float
    outpath: str


@dataclass(frozen=True)
class DatasetDifferenceSummary:
    ticker: str
    split: str
    plot_type: str
    rows: int
    anomalies: int
    mean_pct_diff: float
    median_pct_diff: float
    min_pct_diff: float
    max_pct_diff: float
    outpath: str


@dataclass(frozen=True)
class ModelPanelSummary:
    ticker: str
    split: str
    rows: int
    score_windows: int
    highlighted_days: int
    log_return_mean: float
    log_return_std: float
    upper_3std: float
    lower_3std: float
    outpath: str


def parse_csv_list(value: str | None, default: Iterable[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_attention_index(value, size: int, name: str) -> int:
    if isinstance(value, str) and value.lower() == "last":
        return size - 1
    index = int(value)
    if index < 0:
        index = size + index
    if index < 0 or index >= size:
        raise ValueError(f"{name} index out of range: {value} for size {size}")
    return index


def _resolve_attention_indices(value, size: int, name: str) -> list[int]:
    if isinstance(value, str) and value.lower() == "all":
        return list(range(size))
    return [_resolve_attention_index(value, size, name)]


def _date_tick_positions(dates: np.ndarray, count: int = 6) -> tuple[np.ndarray, list[str]]:
    length = len(dates)
    if length == 0:
        return np.asarray([], dtype=int), []
    positions = np.linspace(0, length - 1, num=min(count, length), dtype=int)
    labels = [str(pd.to_datetime(dates[pos]).date()) for pos in positions]
    return positions, labels


def _mask_attention_diagonal(matrix: np.ndarray) -> np.ndarray:
    masked = matrix.astype(float).copy()
    diagonal_len = min(masked.shape[-2], masked.shape[-1])
    idx = np.arange(diagonal_len)
    masked[idx, idx] = np.nan
    return masked


def _finite_max(*arrays: np.ndarray) -> float:
    values = []
    for array in arrays:
        finite = np.asarray(array, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values.append(float(np.max(finite)))
    return max(values) if values else 1e-8


def _with_log_return(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty or "log_return" in frame.columns:
        return frame
    out = frame.copy()
    if "LogReturn" in out.columns:
        out["log_return"] = pd.to_numeric(out["LogReturn"], errors="coerce")
    elif "Close" in out.columns:
        close = pd.to_numeric(out["Close"], errors="coerce")
        out["log_return"] = np.log(close / close.shift(1))
    return out


def plot_attention_window(
    manifest_row: pd.Series,
    outdir: Path,
    layer: int | str = 0,
    head: int | str = 0,
    context_frame: pd.DataFrame | None = None,
    price_dir: Path | str | None = None,
    mask_diagonal: bool = False,
    endpoint_profile_ymax: float | None = None,
    endpoint_diff_ymax: float | None = None,
) -> AttentionPlotSummary:
    data_path = Path(str(manifest_row["attention_npz"]))
    if not data_path.exists():
        raise FileNotFoundError(f"Missing attention artifact: {data_path}")
    payload = np.load(data_path, allow_pickle=False)
    series_all = payload["series"]
    prior_all = payload["prior"]
    dates = payload["dates"]
    association_mode = str(manifest_row.get("association_mode", "GBM"))
    prior_title = f"{association_mode} prior attention"
    layer_idx = _resolve_attention_index(layer, series_all.shape[0], "layer")
    head_idx = _resolve_attention_index(head, series_all.shape[1], "head")
    series = series_all[layer_idx, head_idx]
    prior = prior_all[layer_idx, head_idx]
    diff_abs = np.abs(series - prior)
    series_plot = _mask_attention_diagonal(series) if mask_diagonal else series
    prior_plot = _mask_attention_diagonal(prior) if mask_diagonal else prior
    diff_plot = _mask_attention_diagonal(diff_abs) if mask_diagonal else diff_abs
    endpoint_series = series[-1].astype(float).copy()
    endpoint_prior = prior[-1].astype(float).copy()
    if mask_diagonal and len(endpoint_series):
        endpoint_series[-1] = np.nan
        endpoint_prior[-1] = np.nan
    endpoint_diff = np.abs(endpoint_series - endpoint_prior)

    outdir.mkdir(parents=True, exist_ok=True)
    ticker = str(manifest_row["ticker"])
    window_id = int(manifest_row["window_id"])
    suffix = "_no_diag" if mask_diagonal else ""
    outpath = outdir / f"{ticker}_window{window_id}_layer{layer_idx}_head{head_idx}{suffix}.png"

    fig = plt.figure(figsize=(15, 14))
    grid = fig.add_gridspec(5, 3, height_ratios=[0.55, 0.55, 0.55, 1.0, 0.75])
    ax_close = fig.add_subplot(grid[0, :])
    ax_log_return = fig.add_subplot(grid[1, :], sharex=ax_close)
    ax_score = fig.add_subplot(grid[2, :], sharex=ax_close)
    ax_series = fig.add_subplot(grid[3, 0])
    ax_prior = fig.add_subplot(grid[3, 1])
    ax_diff = fig.add_subplot(grid[3, 2])
    ax_profile = fig.add_subplot(grid[4, 0])
    ax_delta = fig.add_subplot(grid[4, 1])
    ax_text = fig.add_subplot(grid[4, 2])

    event_date = pd.to_datetime(manifest_row.get("end_date", ""))
    reference_spike_date_raw = manifest_row.get("reference_spike_end_date", "")
    reference_spike_date = pd.to_datetime(reference_spike_date_raw) if str(reference_spike_date_raw) else pd.NaT
    resolved_price_dir = Path(price_dir) if price_dir is not None else DEFAULT_PRICE_DIR
    if context_frame is not None and not context_frame.empty:
        ticker_context = context_frame[context_frame["ticker"].astype(str) == ticker].copy()
        ticker_context["end_date"] = pd.to_datetime(ticker_context["end_date"])
        ticker_context = ticker_context.sort_values("end_date")
    else:
        ticker_context = pd.DataFrame()

    price_frame = None
    if not ticker_context.empty:
        start_date = ticker_context["end_date"].min()
        end_date = ticker_context["end_date"].max()
        price_frame = load_price_frame(ticker, resolved_price_dir, start_date, end_date)
    price_frame = _with_log_return(price_frame)
    if price_frame is not None and not price_frame.empty:
        ax_close.plot(price_frame["Date"], price_frame["Close"], color="#111111", linewidth=1.0)
        marker_rows = price_frame[pd.to_datetime(price_frame["Date"]) == event_date]
        if not marker_rows.empty:
            ax_close.scatter(
                marker_rows["Date"],
                marker_rows["Close"],
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                s=80,
                zorder=5,
            )
    else:
        ax_close.plot([event_date], [0.0], alpha=0.0)
    ax_close.axvline(event_date, color="red", alpha=0.65, linewidth=1.2)
    if pd.notna(reference_spike_date):
        ax_close.axvline(reference_spike_date, color="#ff7f0e", alpha=0.75, linewidth=1.1, linestyle="--")
    ax_close.set_title("Close with selected attention point")
    ax_close.set_ylabel("Close")
    ax_close.grid(True, alpha=0.25)

    if price_frame is not None and not price_frame.empty and "log_return" in price_frame.columns:
        log_return = pd.to_numeric(price_frame["log_return"], errors="coerce")
        ax_log_return.plot(price_frame["Date"], log_return, color="#1f77b4", linewidth=1.0, label="log_return")
        ax_log_return.axhline(0.0, color="#222222", linewidth=0.8)
        clean_log_return = log_return.dropna()
        if not clean_log_return.empty:
            mean_lr = float(clean_log_return.mean())
            std_lr = float(clean_log_return.std(ddof=0))
            ax_log_return.axhline(mean_lr + 3.0 * std_lr, color="#9467bd", linestyle="-.", linewidth=1.0, label="+3 std")
            ax_log_return.axhline(mean_lr - 3.0 * std_lr, color="#9467bd", linestyle="-.", linewidth=1.0, label="-3 std")
        marker_rows = price_frame[pd.to_datetime(price_frame["Date"]) == event_date]
        if not marker_rows.empty and "log_return" in marker_rows.columns:
            ax_log_return.scatter(
                marker_rows["Date"],
                pd.to_numeric(marker_rows["log_return"], errors="coerce"),
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                s=70,
                zorder=5,
            )
    else:
        ax_log_return.plot([event_date], [0.0], alpha=0.0)
    ax_log_return.axvline(event_date, color="red", alpha=0.65, linewidth=1.2)
    if pd.notna(reference_spike_date):
        ax_log_return.axvline(reference_spike_date, color="#ff7f0e", alpha=0.75, linewidth=1.1, linestyle="--")
    ax_log_return.set_title("Log return")
    ax_log_return.set_ylabel("log_return")
    ax_log_return.grid(True, alpha=0.25)
    handles, labels = ax_log_return.get_legend_handles_labels()
    if handles:
        ax_log_return.legend(handles, labels, loc="upper left")

    if not ticker_context.empty and "score" in ticker_context.columns:
        ax_score.plot(
            ticker_context["end_date"],
            pd.to_numeric(ticker_context["score"], errors="coerce"),
            color="#8c564b",
            linewidth=1.0,
            label="anomaly score",
        )
        marker_context = ticker_context[ticker_context["end_date"] == event_date]
        if not marker_context.empty:
            marker_score = pd.to_numeric(marker_context["score"], errors="coerce")
            ax_score.scatter(
                marker_context["end_date"],
                marker_score,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                s=80,
                zorder=5,
                label="attention window",
            )
    ax_score.axvline(event_date, color="red", alpha=0.65, linewidth=1.2)
    if pd.notna(reference_spike_date):
        ax_score.axvline(
            reference_spike_date,
            color="#ff7f0e",
            alpha=0.75,
            linewidth=1.1,
            linestyle="--",
            label="reference spike",
        )
    ax_score.set_title("Anomaly score")
    ax_score.set_ylabel("score")
    ax_score.grid(True, alpha=0.25)
    handles, labels = ax_score.get_legend_handles_labels()
    if handles:
        ax_score.legend(handles, labels, loc="upper left")

    if not ticker_context.empty and "true_log_return_anomaly" in ticker_context.columns:
        event_dates = pd.to_datetime(
            ticker_context.loc[
                pd.to_numeric(ticker_context["true_log_return_anomaly"], errors="coerce").fillna(0).astype(int) > 0,
                "end_date",
            ]
        )
        _highlight_dates([ax_close, ax_log_return, ax_score], event_dates, color="green", alpha=0.12)

    vmax = max(_finite_max(series_plot, prior_plot), 1e-8)
    diff_vmax = max(_finite_max(diff_plot), 1e-8)
    title_suffix = " (diag masked)" if mask_diagonal else ""
    image_specs = [
        (ax_series, series_plot, f"Learned series attention{title_suffix}", vmax, "viridis"),
        (ax_prior, prior_plot, f"{prior_title}{title_suffix}", vmax, "viridis"),
        (ax_diff, diff_plot, f"|series - prior|{title_suffix}", diff_vmax, "magma"),
    ]
    for ax, matrix, title, max_value, cmap in image_specs:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#f2f2f2")
        im = ax.imshow(matrix, aspect="auto", origin="upper", vmin=0.0, vmax=max_value, cmap=cmap_obj)
        ax.set_title(title)
        ax.set_xlabel("Key date")
        ax.set_ylabel("Query date")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_positions, tick_labels = _date_tick_positions(dates)
    for ax in (ax_series, ax_prior, ax_diff):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)

    x = np.arange(len(endpoint_series))
    ax_profile.plot(x, endpoint_series, label="series endpoint", color="#1f77b4", linewidth=1.4)
    ax_profile.plot(x, endpoint_prior, label="prior endpoint", color="#ff7f0e", linewidth=1.2)
    ax_profile.set_title("Endpoint attention row")
    ax_profile.set_xlabel("Key position")
    ax_profile.set_ylabel("Weight")
    if endpoint_profile_ymax is not None and np.isfinite(endpoint_profile_ymax) and endpoint_profile_ymax > 0:
        ax_profile.set_ylim(0.0, float(endpoint_profile_ymax))
    ax_profile.grid(True, alpha=0.25)
    ax_profile.legend(loc="upper left")

    ax_delta.bar(x, np.nan_to_num(endpoint_diff, nan=0.0), color="#8c564b", width=0.9)
    ax_delta.set_title("Endpoint absolute difference")
    ax_delta.set_xlabel("Key position")
    ax_delta.set_ylabel("Abs diff")
    if endpoint_diff_ymax is not None and np.isfinite(endpoint_diff_ymax) and endpoint_diff_ymax > 0:
        ax_delta.set_ylim(0.0, float(endpoint_diff_ymax))
    ax_delta.grid(True, axis="y", alpha=0.25)

    ax_text.axis("off")
    if np.isfinite(endpoint_series).any():
        endpoint_top_index = int(np.nanargmax(endpoint_series))
        top_lag = int(len(endpoint_series) - 1 - endpoint_top_index)
        top_weight = float(endpoint_series[endpoint_top_index])
    else:
        top_lag = -1
        top_weight = float("nan")
    endpoint_l1 = float(np.nansum(endpoint_diff))
    score_formula = str(manifest_row.get("score_formula", "sum"))
    score_term_label = "nll*assoc" if score_formula == "product" else "nll+assoc"
    raw_sum = float(manifest_row.get("raw_sum", np.nan))
    score_z = float(manifest_row.get("score_robust_z", np.nan))
    prev_score_z = float(manifest_row.get("prev_score_robust_z", np.nan))
    delta_score_z = float(manifest_row.get("delta_score_robust_z", np.nan))
    reference_spike_window_id = int(manifest_row.get("reference_spike_window_id", -1))
    reference_spike_delta = float(manifest_row.get("reference_spike_delta_score_robust_z", np.nan))
    reference_spike_z = float(manifest_row.get("reference_spike_score_robust_z", np.nan))
    selection_role = str(manifest_row.get("selection_role", "selected"))
    description = (
        f"Ticker: {ticker}\n"
        f"Window: {window_id}\n"
        f"Selection role: {selection_role}\n"
        f"Dates: {manifest_row.get('start_date', '')} to {manifest_row.get('end_date', '')}\n"
        f"Reference spike: {reference_spike_window_id} @ {reference_spike_date_raw}\n"
        f"Reference spike delta z: {reference_spike_delta:.6g}\n"
        f"Reference spike score z: {reference_spike_z:.6g}\n"
        f"Layer/head: {layer_idx}/{head_idx}\n"
        f"Diagonal masked: {mask_diagonal}\n"
        f"Association mode: {association_mode}\n"
        f"Score: {float(manifest_row.get('score', np.nan)):.6g}\n"
        f"Raw {score_term_label}: {raw_sum:.6g}\n"
        f"Score z prev/current: {prev_score_z:.6g} -> {score_z:.6g}\n"
        f"Delta score z: {delta_score_z:.6g}\n"
        f"Association discrepancy: {float(manifest_row.get('association_discrepancy', np.nan)):.6g}\n"
        f"Endpoint top lag: {top_lag}\n"
        f"Endpoint top weight: {top_weight:.6g}\n"
        f"Endpoint L1 gap: {endpoint_l1:.6g}\n\n"
        "Interpretation:\n"
        "Series is the transformer's learned attention.\n"
        "Prior is the configured GBM timestamp posterior.\n"
        "Large gaps mark departures from that prior.\n"
        "When diagonal is masked, self-attention is excluded from the displayed scale."
    )
    ax_text.text(0.0, 1.0, description, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(f"{ticker} attention diagnostic | window {window_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return AttentionPlotSummary(
        ticker=ticker,
        window_id=window_id,
        layer=layer_idx,
        head=head_idx,
        endpoint_top_lag=top_lag,
        endpoint_top_weight=top_weight,
        endpoint_l1_mean=endpoint_l1,
        outpath=str(outpath),
    )


def _attention_endpoint_axis_limits(
    manifest: pd.DataFrame,
    layer: int | str,
    head: int | str,
    mask_diagonal: bool,
) -> tuple[float, float]:
    profile_max = 0.0
    diff_max = 0.0
    for _, row in manifest.iterrows():
        payload = np.load(Path(str(row["attention_npz"])), allow_pickle=False)
        try:
            layer_indices = _resolve_attention_indices(layer, payload["series"].shape[0], "layer")
            head_indices = _resolve_attention_indices(head, payload["series"].shape[1], "head")
            for layer_idx in layer_indices:
                for head_idx in head_indices:
                    series = payload["series"][layer_idx, head_idx]
                    prior = payload["prior"][layer_idx, head_idx]
                    endpoint_series = series[-1].astype(float).copy()
                    endpoint_prior = prior[-1].astype(float).copy()
                    if mask_diagonal and len(endpoint_series):
                        endpoint_series[-1] = np.nan
                        endpoint_prior[-1] = np.nan
                    endpoint_diff = np.abs(endpoint_series - endpoint_prior)
                    profile_max = max(profile_max, _finite_max(endpoint_series, endpoint_prior))
                    diff_max = max(diff_max, _finite_max(endpoint_diff))
        finally:
            payload.close()
    return max(profile_max, 1e-8), max(diff_max, 1e-8)


def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return pd.Series(np.zeros(len(values)), index=values.index)

    median = float(clean.median())
    mad = float((clean - median).abs().median())
    scale = MAD_SCALE * mad
    if scale <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - median) / scale


def mad_threshold(series: pd.Series, k: float) -> MadThreshold:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return MadThreshold(float("nan"), float("nan"), float("nan"))

    median = float(values.median())
    mad = float((values - median).abs().median())
    scale = MAD_SCALE * mad
    if scale <= 1e-12:
        threshold = float(values.mean() + k * values.std(ddof=0))
    else:
        threshold = median + k * scale
    return MadThreshold(threshold=threshold, median=median, mad=mad)


def add_final_score_columns(frame: pd.DataFrame, score_formula: str = "sum") -> pd.DataFrame:
    required = {"nll", "association_discrepancy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required score columns: {', '.join(missing)}")

    out = frame.copy()
    out["nll"] = pd.to_numeric(out["nll"], errors="coerce")
    out["association_discrepancy"] = pd.to_numeric(out["association_discrepancy"], errors="coerce")
    score_formula = str(score_formula or "sum").lower()
    if score_formula == "sum":
        out["raw_sum"] = out["nll"].fillna(0) + out["association_discrepancy"].fillna(0)
    elif score_formula == "product":
        out["raw_sum"] = out["nll"].fillna(0) * out["association_discrepancy"].fillna(0)
    elif score_formula in {"softmax_product", "nll_assoc_softmax_product"}:
        if "softmax_product_score" not in out.columns:
            raise ValueError("score_formula=softmax_product requires softmax_product_score in the score CSV")
        out["raw_sum"] = pd.to_numeric(out["softmax_product_score"], errors="coerce")
        score_formula = "softmax_product"
    elif score_formula in {"mle_param_softmax_product", "mle_param"}:
        if "mle_param_softmax_product_score" not in out.columns:
            raise ValueError("score_formula=mle_param_softmax_product requires mle_param_softmax_product_score in the score CSV")
        out["raw_sum"] = pd.to_numeric(out["mle_param_softmax_product_score"], errors="coerce")
        score_formula = "mle_param_softmax_product"
    else:
        raise ValueError("score_formula must be sum, product, softmax_product, or mle_param_softmax_product")
    out["score_formula"] = score_formula
    out["score_raw"] = out["raw_sum"]
    out["delta_score_raw"] = out["score_raw"].diff().abs()
    out["score_robust_z"] = robust_z(out["raw_sum"])
    out["delta_score_robust_z"] = out["score_robust_z"].diff().abs()
    return out


def final_spike_mask(frame: pd.DataFrame, k: float, threshold_mode: str = "delta") -> tuple[pd.Series, MadThreshold]:
    if threshold_mode == "raw":
        stats = mad_threshold(frame["score_raw"], k=k)
        return frame["score_raw"] > stats.threshold, stats
    if threshold_mode == "delta":
        stats = mad_threshold(frame["delta_score_raw"], k=k)
        return frame["delta_score_raw"] > stats.threshold, stats
    raise ValueError("threshold_mode must be raw or delta")


def load_price_frame(ticker: str, price_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    path = price_dir / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return None

    price_df = pd.read_csv(path, parse_dates=["Date"])
    label_path = price_dir / f"{ticker}_anomaly_label.csv"
    if label_path.exists():
        labels = pd.read_csv(label_path)
        if "Date" not in labels.columns:
            labels = labels.rename(columns={labels.columns[0]: "Date"})
        labels["Date"] = pd.to_datetime(labels["Date"])
        price_df = price_df.merge(labels, on="Date", how="left")
    price_df = price_df[(price_df["Date"] >= start) & (price_df["Date"] <= end)].copy()
    if price_df.empty:
        return None
    return price_df.sort_values("Date").reset_index(drop=True)


def price_anomaly_starts(
    price_df: pd.DataFrame | None,
    scored_dates: set[pd.Timestamp],
    price_z_thr: float,
) -> list[pd.Timestamp]:
    if price_df is None or price_df.empty:
        return []

    frame = price_df.copy()
    frame["ret"] = frame["Close"].pct_change()
    frame["ret_z"] = robust_z(frame["ret"])
    frame["is_anom"] = frame["ret_z"].abs() > price_z_thr

    starts: list[pd.Timestamp] = []
    previous = False
    for _, row in frame.iterrows():
        current = bool(row["is_anom"])
        date = pd.to_datetime(row["Date"])
        if current and not previous and date in scored_dates:
            starts.append(date)
        previous = current
    return starts


MAD_K = 9.0
DEFAULT_LABEL_COLUMNS = ["jump", "drop", "volatility_shock"]
DEFAULT_PRICE_DIR = Path("datasets/SP500")


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return token.strip("._") or "ticker"


def _discover_dataset_tickers(data_path: Path) -> list[str]:
    tickers = []
    for path in sorted(data_path.glob("*_ohlcv.csv")):
        ticker = path.stem.replace("_ohlcv", "")
        if (data_path / f"{ticker}_anomaly_label.csv").exists():
            tickers.append(ticker)
    return tickers


def _load_dataset_ticker_frame(data_path: Path, ticker: str) -> pd.DataFrame:
    ohlcv_path = data_path / f"{ticker}_ohlcv.csv"
    label_path = data_path / f"{ticker}_anomaly_label.csv"
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
    return ohlcv.merge(labels, on="Date", how="inner").sort_values("Date").reset_index(drop=True)


def _add_close_difference_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["close_diff_pct"] = out["Close"].pct_change() * 100.0
    label_columns = [col for col in detect_label_columns(out) if col != "Date"]
    if label_columns:
        out["is_anomaly"] = (out[label_columns].fillna(0).to_numpy() > 0).any(axis=1)
    else:
        out["is_anomaly"] = False
    return out


def _chronological_endpoint_split_indices(
    n_rows: int,
    window_size: int,
    step: int,
    split_ratios: tuple[float, float, float],
    purge_gap: int | None,
) -> dict[str, set[int]]:
    starts = list(range(0, max(1, n_rows - window_size + 1), step))
    if n_rows < window_size:
        starts = [0]
    effective_gap = int(np.ceil((window_size - 1) / max(1, step))) if purge_gap is None else max(0, int(purge_gap))
    n_total = len(starts)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    train_end = n_train
    val_start = min(n_total, train_end + effective_gap)
    val_end = min(n_total, val_start + n_val)
    test_start = min(n_total, val_end + effective_gap)
    split_starts = {
        "train": starts[:train_end],
        "test": starts[test_start:],
    }
    return {
        split: {min(start + window_size - 1, n_rows - 1) for start in split_values}
        for split, split_values in split_starts.items()
    }


def _plot_close_difference_histogram(
    values: pd.Series,
    ticker: str,
    split: str,
    plot_type: str,
    outpath: Path,
    anomaly_count: int,
) -> DatasetDifferenceSummary:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if clean.empty:
        ax.text(0.5, 0.5, "No valid close differences", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(-1, 1)
    else:
        ax.hist(clean, bins=50, color="#4c78a8", edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color="#222222", linewidth=1.0)
        ax.axvline(float(clean.median()), color="#f58518", linestyle="--", linewidth=1.2, label=f"median={clean.median():.4g}%")
        ax.legend(loc="upper right")
    label = "anomaly days" if plot_type == "anomaly" else "all endpoint days"
    ax.set_title(f"{ticker} {split} | Close(t) vs Close(t-1) % difference | {label}")
    ax.set_xlabel("Close difference (%)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return DatasetDifferenceSummary(
        ticker=ticker,
        split=split,
        plot_type=plot_type,
        rows=int(len(clean)),
        anomalies=int(anomaly_count),
        mean_pct_diff=float(clean.mean()) if not clean.empty else float("nan"),
        median_pct_diff=float(clean.median()) if not clean.empty else float("nan"),
        min_pct_diff=float(clean.min()) if not clean.empty else float("nan"),
        max_pct_diff=float(clean.max()) if not clean.empty else float("nan"),
        outpath=str(outpath),
    )


def run_dataset_difference_histograms(args) -> pd.DataFrame:
    data_path = Path(getattr(args, "data_path", "datasets/SP500_event_taxonomy_w100"))
    window_size = int(getattr(args, "window_size", 100))
    step = int(getattr(args, "step", 1))
    split_method = str(getattr(args, "split_method", "chronological") or "chronological")
    if split_method != "chronological":
        raise ValueError("dataset_difference_histograms currently supports chronological split only")
    split_ratios_value = getattr(args, "split_ratios", None)
    if split_ratios_value is None:
        split_ratios = (0.7, 0.15, 0.15)
    elif isinstance(split_ratios_value, str):
        split_ratios = tuple(float(item.strip()) for item in split_ratios_value.split(",") if item.strip())
    else:
        split_ratios = tuple(float(item) for item in split_ratios_value)
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train,val,test values")
    purge_gap = getattr(args, "purge_gap", None)
    histogram_all_dir = Path(getattr(args, "histogram_all_out"))
    histogram_anomaly_dir = Path(getattr(args, "histogram_anomaly_out"))

    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    tickers = requested_tickers or _discover_dataset_tickers(data_path)
    if not tickers:
        raise ValueError(f"No tickers found in {data_path}")

    summaries: list[DatasetDifferenceSummary] = []
    for index, ticker in enumerate(tickers):
        frame = _add_close_difference_columns(_load_dataset_ticker_frame(data_path, ticker))
        split_indices = _chronological_endpoint_split_indices(
            n_rows=len(frame),
            window_size=window_size,
            step=step,
            split_ratios=split_ratios,
            purge_gap=purge_gap,
        )
        file_ticker = _safe_filename_token(ticker)
        for split in ("train", "test"):
            split_mask = frame.index.to_series().isin(split_indices[split])
            split_frame = frame.loc[split_mask].copy()
            anomaly_frame = split_frame[split_frame["is_anomaly"]].copy()
            summaries.append(
                _plot_close_difference_histogram(
                    split_frame["close_diff_pct"],
                    ticker=ticker,
                    split=split,
                    plot_type="all",
                    outpath=histogram_all_dir / f"{index:03d}_{file_ticker}_{split}.png",
                    anomaly_count=int(split_frame["is_anomaly"].sum()),
                )
            )
            summaries.append(
                _plot_close_difference_histogram(
                    anomaly_frame["close_diff_pct"],
                    ticker=ticker,
                    split=split,
                    plot_type="anomaly",
                    outpath=histogram_anomaly_dir / f"{index:03d}_{file_ticker}_anomaly_{split}.png",
                    anomaly_count=int(len(anomaly_frame)),
                )
            )
        if (index + 1) % max(1, len(tickers) // 10) == 0:
            print(f"[{index + 1}/{len(tickers)}] processed {ticker}", flush=True)

    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    histogram_all_dir.mkdir(parents=True, exist_ok=True)
    histogram_anomaly_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(histogram_all_dir / "close_difference_histogram_summary.csv", index=False)
    summary_df.to_csv(histogram_anomaly_dir / "close_difference_histogram_summary.csv", index=False)
    print(
        "Done. Dataset validation histograms written to "
        f"{histogram_all_dir} and {histogram_anomaly_dir}"
    )
    return summary_df


def _plot_logreturn_histogram(
    values: pd.Series,
    ticker: str,
    plot_type: str,
    outpath: Path,
    anomaly_count: int,
) -> DatasetDifferenceSummary:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if clean.empty:
        ax.text(0.5, 0.5, "No valid log returns", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(-0.01, 0.01)
    else:
        ax.hist(clean, bins=60, color="#4c78a8", edgecolor="white", linewidth=0.5)
        ax.axvline(0.0, color="#222222", linewidth=1.0)
        ax.axvline(float(clean.median()), color="#f58518", linestyle="--", linewidth=1.2, label=f"median={clean.median():.4g}")
        ax.axvline(float(clean.mean()), color="#54a24b", linestyle=":", linewidth=1.2, label=f"mean={clean.mean():.4g}")
        ax.legend(loc="upper right")
    label = "anomaly days" if plot_type == "anomaly" else "all days"
    ax.set_title(f"{ticker} | log_return distribution | {label}")
    ax.set_xlabel("log_return")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return DatasetDifferenceSummary(
        ticker=ticker,
        split="daily",
        plot_type=plot_type,
        rows=int(len(clean)),
        anomalies=int(anomaly_count),
        mean_pct_diff=float(clean.mean()) if not clean.empty else float("nan"),
        median_pct_diff=float(clean.median()) if not clean.empty else float("nan"),
        min_pct_diff=float(clean.min()) if not clean.empty else float("nan"),
        max_pct_diff=float(clean.max()) if not clean.empty else float("nan"),
        outpath=str(outpath),
    )


def run_dataset_logreturn_histograms(args) -> pd.DataFrame:
    data_path = Path(getattr(args, "data_path", "datasets/SP500_logreturn_volume_w60"))
    histogram_all_dir = Path(getattr(args, "histogram_all_out"))
    histogram_anomaly_dir = Path(getattr(args, "histogram_anomaly_out"))
    anomaly_label_column = str(getattr(args, "anomaly_label_column", "log_return_anomaly") or "log_return_anomaly")
    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    tickers = requested_tickers or _discover_dataset_tickers(data_path)
    if not tickers:
        raise ValueError(f"No prepared tickers found in {data_path}")

    summaries: list[DatasetDifferenceSummary] = []
    for index, ticker in enumerate(tickers):
        frame = _load_dataset_ticker_frame(data_path, ticker)
        if "log_return" not in frame.columns:
            raise ValueError(f"{ticker} is missing log_return in {data_path}")
        if anomaly_label_column not in frame.columns:
            raise ValueError(f"{ticker} is missing requested anomaly label column: {anomaly_label_column}")
        anomaly_mask = pd.to_numeric(frame[anomaly_label_column], errors="coerce").fillna(0) > 0
        file_ticker = _safe_filename_token(ticker)
        summaries.append(
            _plot_logreturn_histogram(
                frame["log_return"],
                ticker=ticker,
                plot_type="all",
                outpath=histogram_all_dir / f"{index:03d}_{file_ticker}.png",
                anomaly_count=int(np.asarray(anomaly_mask).sum()),
            )
        )
        summaries.append(
            _plot_logreturn_histogram(
                frame.loc[anomaly_mask, "log_return"],
                ticker=ticker,
                plot_type="anomaly",
                outpath=histogram_anomaly_dir / f"{index:03d}_{file_ticker}_anomaly.png",
                anomaly_count=int(np.asarray(anomaly_mask).sum()),
            )
        )
        if (index + 1) % max(1, len(tickers) // 10) == 0:
            print(f"[{index + 1}/{len(tickers)}] processed {ticker}", flush=True)

    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    histogram_all_dir.mkdir(parents=True, exist_ok=True)
    histogram_anomaly_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(histogram_all_dir / "logreturn_histogram_summary.csv", index=False)
    summary_df.to_csv(histogram_anomaly_dir / "logreturn_histogram_summary.csv", index=False)
    print(f"Done. Log-return histograms written to {histogram_all_dir} and {histogram_anomaly_dir}")
    return summary_df


def _highlight_dates(axes, dates: Iterable[pd.Timestamp], color: str = "green", alpha: float = 0.12) -> None:
    for date in dates:
        start = pd.to_datetime(date) - pd.Timedelta(hours=12)
        end = pd.to_datetime(date) + pd.Timedelta(hours=12)
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0)


def plot_model_diagnostic_panel(
    scores_df: pd.DataFrame,
    data_path: Path,
    ticker: str,
    outdir: Path,
    split: str = "test",
    label_columns: list[str] | None = None,
    score_column: str = "score",
) -> ModelPanelSummary | None:
    frame = _load_dataset_ticker_frame(data_path, ticker)
    if "split" in frame.columns:
        daily = frame[frame["split"].astype(str) == split].copy()
    else:
        daily = frame.copy()
    if daily.empty:
        return None
    ticker_scores = scores_df[scores_df["ticker"].astype(str) == ticker].copy()
    if "split" in ticker_scores.columns:
        ticker_scores = ticker_scores[ticker_scores["split"].astype(str) == split].copy()
    if ticker_scores.empty:
        return None
    ticker_scores["end_date"] = pd.to_datetime(ticker_scores["end_date"])
    ticker_scores[score_column] = pd.to_numeric(ticker_scores[score_column], errors="coerce")
    ticker_scores = ticker_scores.sort_values("end_date")

    labels = label_columns or ["log_return_anomaly", "volume_anomaly"]
    available_labels = [column for column in labels if column in daily.columns]
    if available_labels:
        highlight_mask = (daily[available_labels].fillna(0).astype(float).to_numpy() > 0).any(axis=1)
    else:
        highlight_mask = np.zeros(len(daily), dtype=bool)
    highlight_dates = list(pd.to_datetime(daily.loc[highlight_mask, "Date"]))

    log_return = pd.to_numeric(daily["log_return"], errors="coerce")
    mean_lr = float(log_return.mean())
    std_lr = float(log_return.std(ddof=0))
    upper_3std = mean_lr + 3.0 * std_lr
    lower_3std = mean_lr - 3.0 * std_lr

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{_safe_filename_token(ticker)}_{split}_close_logreturn_score_panel.png"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.85]},
    )
    ax_price, ax_lr, ax_score = axes

    ax_price.plot(daily["Date"], daily["Close"], color="#111111", linewidth=1.1, label="Close")
    ax_price.set_title(f"{ticker} diagnostic | {split} split")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")

    ax_lr.plot(daily["Date"], daily["log_return"], color="#1f77b4", linewidth=1.0, label="log_return")
    ax_lr.axhline(0.0, color="#222222", linewidth=0.9)
    ax_lr.axhline(upper_3std, color="#9467bd", linestyle="-.", linewidth=1.1, label=f"{split} +3 std={upper_3std:.4f}")
    ax_lr.axhline(lower_3std, color="#9467bd", linestyle="-.", linewidth=1.1, label=f"{split} -3 std={lower_3std:.4f}")
    ax_lr.set_ylabel("log_return")
    ax_lr.grid(True, alpha=0.25)
    ax_lr.legend(loc="upper left")

    ax_score.plot(ticker_scores["end_date"], ticker_scores[score_column], color="#8c564b", linewidth=1.0, label="anomaly score")
    ax_score.set_ylabel("score")
    ax_score.set_xlabel("Date")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="upper left")

    _highlight_dates(axes, highlight_dates, color="green", alpha=0.12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return ModelPanelSummary(
        ticker=ticker,
        split=split,
        rows=int(len(daily)),
        score_windows=int(len(ticker_scores)),
        highlighted_days=int(len(highlight_dates)),
        log_return_mean=mean_lr,
        log_return_std=std_lr,
        upper_3std=upper_3std,
        lower_3std=lower_3std,
        outpath=str(outpath),
    )


def run_model_diagnostic_panels(args) -> pd.DataFrame:
    scores_df = pd.read_csv(args.csv, parse_dates=["end_date"])
    data_path = Path(getattr(args, "data_path"))
    outdir = Path(getattr(args, "out"))
    split = str(getattr(args, "split", "test") or "test")
    score_column = str(getattr(args, "score_column", "score") or "score")
    label_columns = parse_csv_list(getattr(args, "label_names", None), ["log_return_anomaly", "volume_anomaly"])
    tickers = sorted(scores_df["ticker"].astype(str).unique())
    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    if requested_tickers:
        requested = set(requested_tickers)
        tickers = [ticker for ticker in tickers if ticker in requested]
    summaries = []
    for idx, ticker in enumerate(tickers, start=1):
        result = plot_model_diagnostic_panel(
            scores_df=scores_df,
            data_path=data_path,
            ticker=ticker,
            outdir=outdir,
            split=split,
            label_columns=label_columns,
            score_column=score_column,
        )
        if result is not None:
            summaries.append(result)
        if idx % max(1, len(tickers) // 10) == 0:
            print(f"[{idx}/{len(tickers)}] panel processed {ticker}", flush=True)
    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "model_diagnostic_panel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Done. Model diagnostic panels in {outdir} summary: {summary_path}")
    return summary_df


def plot_endpoint_logreturn_score_panel(
    scores_df: pd.DataFrame,
    data_path: Path,
    ticker: str,
    outdir: Path,
    split: str = "test",
    base_score_column: str = "score",
    robust_score_column: str = "nll_robust_softmax_assoc_score",
) -> ModelPanelSummary | None:
    ohlcv_path = data_path / f"{ticker}_ohlcv.csv"
    if not ohlcv_path.exists():
        return None
    daily = pd.read_csv(ohlcv_path)
    if "Date" not in daily.columns:
        daily = daily.rename(columns={daily.columns[0]: "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    if "log_return" not in daily.columns:
        close = pd.to_numeric(daily["Close"], errors="coerce")
        daily["log_return"] = np.log(close / close.shift(1))

    ticker_scores = scores_df[scores_df["ticker"].astype(str) == ticker].copy()
    if "split" in ticker_scores.columns:
        ticker_scores = ticker_scores[ticker_scores["split"].astype(str) == split].copy()
    if ticker_scores.empty:
        return None
    ticker_scores["end_date"] = pd.to_datetime(ticker_scores["end_date"])
    ticker_scores = ticker_scores.sort_values("end_date")
    scored_dates = set(pd.to_datetime(ticker_scores["end_date"]))
    daily = daily[daily["Date"].isin(scored_dates)].copy()
    if daily.empty:
        return None

    if "endpoint_log_return_upper_3std" in ticker_scores.columns:
        upper_values = pd.to_numeric(ticker_scores["endpoint_log_return_upper_3std"], errors="coerce").dropna()
        lower_values = pd.to_numeric(ticker_scores["endpoint_log_return_lower_3std"], errors="coerce").dropna()
        mean_values = pd.to_numeric(ticker_scores["endpoint_log_return_mean"], errors="coerce").dropna()
        std_values = pd.to_numeric(ticker_scores["endpoint_log_return_std"], errors="coerce").dropna()
        upper_3std = float(upper_values.iloc[0]) if not upper_values.empty else float("nan")
        lower_3std = float(lower_values.iloc[0]) if not lower_values.empty else float("nan")
        mean_lr = float(mean_values.iloc[0]) if not mean_values.empty else float("nan")
        std_lr = float(std_values.iloc[0]) if not std_values.empty else float("nan")
    else:
        log_return = pd.to_numeric(daily["log_return"], errors="coerce")
        mean_lr = float(log_return.mean())
        std_lr = float(log_return.std(ddof=0))
        upper_3std = mean_lr + 3.0 * std_lr
        lower_3std = mean_lr - 3.0 * std_lr

    if "endpoint_log_return_anomaly" in ticker_scores.columns:
        event_dates = list(
            pd.to_datetime(
                ticker_scores.loc[
                    pd.to_numeric(ticker_scores["endpoint_log_return_anomaly"], errors="coerce").fillna(0).astype(int)
                    > 0,
                    "end_date",
                ]
            )
        )
    else:
        log_return = pd.to_numeric(daily["log_return"], errors="coerce")
        event_dates = list(pd.to_datetime(daily.loc[(log_return > upper_3std) | (log_return < lower_3std), "Date"]))

    for column in [base_score_column, robust_score_column]:
        if column not in ticker_scores.columns:
            raise ValueError(f"Missing score column for endpoint panel: {column}")
        ticker_scores[column] = pd.to_numeric(ticker_scores[column], errors="coerce")

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{_safe_filename_token(ticker)}_{split}_endpoint_logreturn_score_panel.png"
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9, 0.8, 0.8]},
    )
    ax_price, ax_lr, ax_base, ax_robust = axes

    ax_price.plot(daily["Date"], daily["Close"], color="#111111", linewidth=1.0, label="Close")
    ax_price.set_title(f"{ticker} | endpoint log_return anomaly | {split}")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left")

    ax_lr.plot(daily["Date"], daily["log_return"], color="#1f77b4", linewidth=1.0, label="log_return")
    ax_lr.axhline(0.0, color="#222222", linewidth=0.9)
    ax_lr.axhline(upper_3std, color="#9467bd", linestyle="-.", linewidth=1.1, label=f"+3 std={upper_3std:.4f}")
    ax_lr.axhline(lower_3std, color="#9467bd", linestyle="-.", linewidth=1.1, label=f"-3 std={lower_3std:.4f}")
    ax_lr.set_ylabel("log_return")
    ax_lr.grid(True, alpha=0.25)
    ax_lr.legend(loc="upper left")

    ax_base.plot(ticker_scores["end_date"], ticker_scores[base_score_column], color="#8c564b", linewidth=1.0, label=base_score_column)
    ax_base.set_ylabel("score")
    ax_base.grid(True, alpha=0.25)
    ax_base.legend(loc="upper left")

    ax_robust.plot(
        ticker_scores["end_date"],
        ticker_scores[robust_score_column],
        color="#2ca02c",
        linewidth=1.0,
        label="nll robust-z * softmax(-association robust-z)",
    )
    ax_robust.set_ylabel("robust score")
    ax_robust.set_xlabel("Date")
    ax_robust.grid(True, alpha=0.25)
    ax_robust.legend(loc="upper left")

    for date in event_dates:
        for ax in axes:
            ax.axvline(pd.to_datetime(date), color="green", alpha=0.28, linewidth=1.1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return ModelPanelSummary(
        ticker=ticker,
        split=split,
        rows=int(len(daily)),
        score_windows=int(len(ticker_scores)),
        highlighted_days=int(len(event_dates)),
        log_return_mean=mean_lr,
        log_return_std=std_lr,
        upper_3std=upper_3std,
        lower_3std=lower_3std,
        outpath=str(outpath),
    )


def run_endpoint_logreturn_score_panels(args) -> pd.DataFrame:
    scores_df = pd.read_csv(args.csv, parse_dates=["end_date"])
    data_path = Path(getattr(args, "data_path"))
    outdir = Path(getattr(args, "out"))
    split = str(getattr(args, "split", "test") or "test")
    base_score_column = str(getattr(args, "base_score_column", "score") or "score")
    robust_score_column = str(
        getattr(args, "robust_score_column", "nll_robust_softmax_assoc_score") or "nll_robust_softmax_assoc_score"
    )
    tickers = sorted(scores_df["ticker"].astype(str).unique())
    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    if requested_tickers:
        requested = set(requested_tickers)
        tickers = [ticker for ticker in tickers if ticker in requested]
    summaries = []
    for idx, ticker in enumerate(tickers, start=1):
        result = plot_endpoint_logreturn_score_panel(
            scores_df=scores_df,
            data_path=data_path,
            ticker=ticker,
            outdir=outdir,
            split=split,
            base_score_column=base_score_column,
            robust_score_column=robust_score_column,
        )
        if result is not None:
            summaries.append(result)
        if idx % max(1, len(tickers) // 10) == 0:
            print(f"[{idx}/{len(tickers)}] endpoint panel processed {ticker}", flush=True)
    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "endpoint_logreturn_score_panel_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Done. Endpoint log-return score panels in {outdir} summary: {summary_path}")
    return summary_df


def add_verticals(axes, dates: Iterable[pd.Timestamp], color: str, alpha: float, linewidth: float, label: str | None = None) -> None:
    shown = False
    for date in dates:
        for ax in axes:
            ax.axvline(date, color=color, alpha=alpha, linewidth=linewidth, label=label if label and not shown else None)
        shown = True


def plot_mad_ticker(
    scores_df: pd.DataFrame,
    ticker: str,
    outdir: Path,
    label_columns: list[str],
    price_dir: Path,
    price_z_thr: float,
    mad_k: float = MAD_K,
    score_formula: str = "sum",
    threshold_mode: str = "delta",
) -> SpikeSummary | None:
    sub = scores_df[scores_df["ticker"].astype(str) == ticker].sort_values("end_date").reset_index(drop=True)
    if sub.empty:
        return None

    required_columns = {"end_date", "nll", "association_discrepancy"}
    missing = sorted(required_columns - set(sub.columns))
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {', '.join(missing)}")

    sub["end_date"] = pd.to_datetime(sub["end_date"])
    sub = add_final_score_columns(sub, score_formula=score_formula)

    threshold_mode = str(threshold_mode or "delta").lower()
    spike_mask, threshold_stats = final_spike_mask(sub, mad_k, threshold_mode=threshold_mode)
    spike_dates = list(sub.loc[spike_mask, "end_date"])
    scored_dates = set(pd.to_datetime(sub["end_date"]))

    start = sub["end_date"].min()
    end = sub["end_date"].max()
    price_df = load_price_frame(ticker, price_dir, start, end)
    price_starts = price_anomaly_starts(price_df, scored_dates, price_z_thr)
    overlap_dates = sorted(set(spike_dates) & set(price_starts))

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{ticker}_mad_spikes.png"

    if threshold_mode == "raw":
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 0.8, 0.9]},
        )
        ax_price, ax_terms, ax_score = axes
        ax_dz = None
    else:
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 0.8, 0.85, 0.75]},
        )
        ax_price, ax_terms, ax_score, ax_dz = axes
    x = sub["end_date"]

    if price_df is not None:
        ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.0)
    else:
        ax_price.plot(x, np.full(len(sub), np.nan), color="#111111")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)

    ax_terms.plot(x, sub["nll"], label="nll", color="#1f77b4")
    ax_terms.plot(x, sub["association_discrepancy"], label="association", color="#ff7f0e")
    if str(score_formula).lower() in {"mle_param_softmax_product", "mle_param"}:
        score_label = "softmax(-association) * MLE parameter error"
    elif str(score_formula).lower() in {"softmax_product", "nll_assoc_softmax_product"}:
        score_label = "softmax(-association) * nll"
    elif str(score_formula).lower() == "product":
        score_label = "nll * association"
    else:
        score_label = "nll + association"
    ax_terms.set_ylabel("Terms")
    ax_terms.legend(loc="upper left")
    ax_terms.grid(True, alpha=0.25)

    ax_score.plot(x, sub["score_raw"], label=f"raw score: {score_label}", color="#8c564b", linewidth=1.1)
    if threshold_mode == "raw":
        ax_score.axhline(
            threshold_stats.threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"k={mad_k:g} MAD raw threshold={threshold_stats.threshold:.3g}",
        )
    ax_score.set_ylabel("Raw score")
    ax_score.legend(loc="upper left")
    ax_score.grid(True, alpha=0.25)

    if ax_dz is not None:
        ax_dz.plot(x, sub["delta_score_raw"], label="abs diff raw score", color="#e377c2")
        ax_dz.axhline(
            threshold_stats.threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"k={mad_k:g} MAD delta threshold={threshold_stats.threshold:.3g}",
        )
        ax_dz.set_ylabel("Abs diff")
        ax_dz.set_xlabel("Date")
        ax_dz.legend(loc="upper left")
        ax_dz.grid(True, alpha=0.25)
    else:
        ax_score.set_xlabel("Date")

    add_verticals(axes, spike_dates, color="red", alpha=0.25, linewidth=0.9)
    for idx, label_column in enumerate(label_columns):
        if label_column not in sub.columns:
            continue
        mask = pd.to_numeric(sub[label_column], errors="coerce").fillna(0) > 0
        add_verticals(axes, list(sub.loc[mask, "end_date"]), color=plt.cm.tab10(idx % 10), alpha=0.6, linewidth=1.0)
    add_verticals(axes, price_starts, color="green", alpha=0.45, linewidth=1.2)
    add_verticals(axes, overlap_dates, color="lime", alpha=0.9, linewidth=1.6)

    fig.suptitle(
        f"{ticker} | score={score_formula} | model spikes={len(spike_dates)}, "
        f"price_anoms={len(price_starts)}, overlaps={len(overlap_dates)} | threshold={threshold_mode}"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return SpikeSummary(
        ticker=ticker,
        model_spikes=len(spike_dates),
        price_anoms=len(price_starts),
        overlaps=len(overlap_dates),
        threshold=threshold_stats.threshold,
        score_median=threshold_stats.median,
        score_mad=threshold_stats.mad,
        threshold_mode=threshold_mode,
        outpath=str(outpath),
    )


def run_mad_visualize(args) -> pd.DataFrame:
    scores_df = pd.read_csv(args.csv, parse_dates=["end_date"])
    label_columns = parse_csv_list(getattr(args, "label_names", None), DEFAULT_LABEL_COLUMNS)
    tickers = sorted(scores_df["ticker"].astype(str).unique())
    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    if requested_tickers:
        requested = set(requested_tickers)
        tickers = [ticker for ticker in tickers if ticker in requested]
    if not tickers:
        raise ValueError("No tickers selected for plotting")

    outdir = Path(args.out)
    price_dir = Path(getattr(args, "price_dir", DEFAULT_PRICE_DIR) or DEFAULT_PRICE_DIR)
    price_z_thr = float(getattr(args, "price_z_thr", 3.0) or 3.0)
    mad_k = float(getattr(args, "outlier_k", MAD_K) or MAD_K)
    score_formula = str(getattr(args, "score_formula", "sum") or "sum")
    threshold_mode = str(getattr(args, "threshold_mode", "delta") or "delta")
    summary: list[SpikeSummary] = []
    for idx, ticker in enumerate(tickers, start=1):
        result = plot_mad_ticker(
            scores_df,
            ticker,
            outdir,
            label_columns,
            price_dir,
            price_z_thr,
            mad_k=mad_k,
            score_formula=score_formula,
            threshold_mode=threshold_mode,
        )
        if result is not None:
            summary.append(result)
        if idx % max(1, len(tickers) // 10) == 0:
            print(f"[{idx}/{len(tickers)}] processed {ticker}", flush=True)

    summary_df = pd.DataFrame([result.__dict__ for result in summary])
    summary_csv = outdir / "final_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Done. Final plots in {outdir} summary: {summary_csv}")
    return summary_df


def run_attention_visualize(args) -> pd.DataFrame:
    manifest_path = Path(args.attention_manifest)
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Attention manifest is empty: {manifest_path}")
    outdir = Path(args.out)
    layer = getattr(args, "attention_layer", 0)
    head = getattr(args, "attention_head", 0)
    context_frame = None
    if "score_context_csv" in manifest.columns:
        context_candidates = [value for value in manifest["score_context_csv"].dropna().astype(str).unique() if value]
        if context_candidates and Path(context_candidates[0]).exists():
            context_frame = pd.read_csv(context_candidates[0])
    price_dir = getattr(args, "price_dir", None)
    mask_diagonal = bool(getattr(args, "attention_mask_diagonal", False))
    fixed_profile_ymax, fixed_diff_ymax = _attention_endpoint_axis_limits(manifest, layer, head, mask_diagonal)
    summaries = []
    for _, row in manifest.iterrows():
        payload = np.load(Path(str(row["attention_npz"])), allow_pickle=False)
        layer_indices = _resolve_attention_indices(layer, payload["series"].shape[0], "layer")
        head_indices = _resolve_attention_indices(head, payload["series"].shape[1], "head")
        payload.close()
        for layer_idx in layer_indices:
            for head_idx in head_indices:
                summaries.append(
                    plot_attention_window(
                        row,
                        outdir,
                        layer=layer_idx,
                        head=head_idx,
                        context_frame=context_frame,
                        price_dir=price_dir,
                        mask_diagonal=mask_diagonal,
                        endpoint_profile_ymax=fixed_profile_ymax,
                        endpoint_diff_ymax=fixed_diff_ymax,
                    )
                )
    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    summary_path = outdir / "attention_plot_summary.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Done. Attention plots in {outdir} summary: {summary_path}")
    return summary_df


def _score_threshold(validation_csv: Path, quantile: float, score_column: str) -> float:
    frame = pd.read_csv(validation_csv)
    if score_column not in frame.columns:
        raise ValueError(f"Missing validation score column: {score_column}")
    return float(pd.to_numeric(frame[score_column], errors="coerce").quantile(quantile))


def _weak_event_intervals(scores: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows = []
    for ticker, sub in scores.sort_values(["ticker", "end_date"]).groupby("ticker"):
        mask = pd.to_numeric(sub[label_column], errors="coerce").fillna(0).astype(int) > 0
        for idx, (start, end) in enumerate(contiguous_intervals(sub["end_date"], mask), start=1):
            event_rows = sub[(sub["end_date"] >= start) & (sub["end_date"] <= end)]
            rows.append(
                {
                    "event_name": f"{ticker}_endpoint_log_return_anomaly_{idx}",
                    "ticker": ticker,
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "event_type": "weak_endpoint_log_return_3std",
                    "days": int(len(event_rows)),
                    "best_score": float(pd.to_numeric(event_rows["score"], errors="coerce").max()),
                }
            )
    return pd.DataFrame(rows)


def _event_metrics(scores: pd.DataFrame, events: pd.DataFrame, threshold: float, score_column: str) -> pd.DataFrame:
    ranked = scores.sort_values(score_column, ascending=False).reset_index(drop=True)
    rank_lookup = {
        (str(row.ticker), pd.to_datetime(row.end_date).date().isoformat()): rank
        for rank, row in enumerate(ranked.itertuples(index=False), start=1)
    }
    rows = []
    for event in events.itertuples(index=False):
        sub = scores[
            (scores["ticker"].astype(str) == str(event.ticker))
            & (scores["end_date"] >= pd.to_datetime(event.start_date))
            & (scores["end_date"] <= pd.to_datetime(event.end_date))
        ].copy()
        if sub.empty:
            continue
        best_idx = pd.to_numeric(sub[score_column], errors="coerce").idxmax()
        best = sub.loc[best_idx]
        best_date = pd.to_datetime(best["end_date"]).date().isoformat()
        rows.append(
            {
                "event_name": event.event_name,
                "ticker": event.ticker,
                "start_date": event.start_date,
                "end_date": event.end_date,
                "detected_at_threshold": bool(float(best[score_column]) > threshold),
                "best_date": best_date,
                "best_score": float(best[score_column]),
                "best_rank": int(rank_lookup.get((str(event.ticker), best_date), -1)),
            }
        )
    return pd.DataFrame(rows)


def _precision_at_top_fraction(scores: pd.DataFrame, label_column: str, score_column: str, fraction: float) -> float:
    k = max(1, int(np.ceil(len(scores) * fraction)))
    top = scores.sort_values(score_column, ascending=False).head(k)
    return float(pd.to_numeric(top[label_column], errors="coerce").fillna(0).mean())


def _draw_weak_event_spans(ax, sub: pd.DataFrame, label_column: str) -> None:
    mask = pd.to_numeric(sub[label_column], errors="coerce").fillna(0).astype(int) > 0
    for start, end in contiguous_intervals(sub["end_date"], mask):
        ax.axvspan(start, interval_end(sub["end_date"], end), color="green", alpha=0.12)


def _ticker_price_frame(data_path: Path, ticker: str, dates: pd.Series) -> pd.DataFrame:
    path = data_path / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return pd.DataFrame()
    daily = pd.read_csv(path)
    if "Date" not in daily.columns:
        daily = daily.rename(columns={daily.columns[0]: "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date")
    if "log_return" not in daily.columns:
        close = pd.to_numeric(daily["Close"], errors="coerce")
        daily["log_return"] = np.log(close / close.shift(1))
    date_set = set(pd.to_datetime(dates))
    return daily[daily["Date"].isin(date_set)].copy()


def _plot_ticker_nextday_figures(
    scores: pd.DataFrame,
    data_path: Path,
    outdir: Path,
    ticker: str,
    threshold: float,
    label_column: str,
    score_column: str,
) -> list[str]:
    paths = []
    sub = scores[scores["ticker"].astype(str) == ticker].sort_values("end_date").copy()
    if sub.empty:
        return paths
    daily = _ticker_price_frame(data_path, ticker, sub["end_date"])
    merged = sub.merge(daily[["Date", "Close", "log_return"]], left_on="end_date", right_on="Date", how="left")
    merged["plot_return"] = pd.to_numeric(merged.get("endpoint_log_return", merged.get("log_return")), errors="coerce")
    merged["realized_vol_20"] = merged["plot_return"].rolling(20, min_periods=5).std(ddof=0).shift(1)
    z = (pd.to_numeric(merged["target_return"], errors="coerce") - pd.to_numeric(merged["mu_pred"], errors="coerce")) / pd.to_numeric(
        merged["sigma_pred"], errors="coerce"
    )
    merged["standardized_residual"] = z.replace([np.inf, -np.inf], np.nan)

    ticker_dir = outdir / "figures" / _safe_filename_token(ticker)
    ticker_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(merged["end_date"], merged["Close"], color="#111111", linewidth=1.0, label="Close")
    axes[1].plot(merged["end_date"], merged["plot_return"], color="#1f77b4", linewidth=1.0, label="log_return")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[2].plot(merged["end_date"], merged[score_column], color="#8c564b", linewidth=1.0, label="NLL score")
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"val p99={threshold:.4g}")
    for ax in axes:
        _draw_weak_event_spans(ax, merged, label_column)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[0].set_title(f"Figure 1 | {ticker} time series with Student-t NLL score")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    path = ticker_dir / "figure1_timeseries_score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    top_k = max(1, int(np.ceil(len(merged) * 0.01)))
    top_mask = merged[score_column].rank(method="first", ascending=False) <= top_k
    hit_mask = top_mask & (pd.to_numeric(merged[label_column], errors="coerce").fillna(0).astype(int) > 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged["end_date"], merged["plot_return"], color="#1f77b4", linewidth=1.0, label="log_return")
    ax.scatter(merged.loc[top_mask & ~hit_mask, "end_date"], merged.loc[top_mask & ~hit_mask, "plot_return"], s=28, color="#d62728", label="Top1% unknown")
    ax.scatter(merged.loc[hit_mask, "end_date"], merged.loc[hit_mask, "plot_return"], s=36, color="#2ca02c", label="Top1% weak-event hit")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_title(f"Figure 2 | {ticker} top anomaly days on return chart")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = ticker_dir / "figure2_top_anomaly_days.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged["end_date"], merged["standardized_residual"], color="#9467bd", linewidth=1.0, label="(r-mu)/sigma")
    for value in [2, 3, 5, -2, -3, -5]:
        ax.axhline(value, color="red" if abs(value) >= 5 else "#666666", linestyle="--", linewidth=0.8)
    _draw_weak_event_spans(ax, merged, label_column)
    ax.set_title(f"Figure 4 | {ticker} standardized residual")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = ticker_dir / "figure4_standardized_residual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(merged["end_date"], merged["sigma_pred"], color="#ff7f0e", linewidth=1.0, label="predicted sigma")
    ax.plot(merged["end_date"], merged["realized_vol_20"], color="#1f77b4", linewidth=1.0, label="rolling vol 20")
    _draw_weak_event_spans(ax, merged, label_column)
    ax.set_title(f"Figure 5 | {ticker} predicted volatility vs realized volatility")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = ticker_dir / "figure5_predicted_vs_realized_volatility.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    event_rows = merged[pd.to_numeric(merged[label_column], errors="coerce").fillna(0).astype(int) > 0]
    if not event_rows.empty:
        center = pd.to_datetime(event_rows.sort_values(score_column, ascending=False).iloc[0]["end_date"])
    else:
        center = pd.to_datetime(merged.sort_values(score_column, ascending=False).iloc[0]["end_date"])
    zoom = merged[(merged["end_date"] >= center - pd.Timedelta(days=45)) & (merged["end_date"] <= center + pd.Timedelta(days=45))]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(zoom["end_date"], zoom["Close"], color="#111111", label="Close")
    axes[1].plot(zoom["end_date"], zoom["plot_return"], color="#1f77b4", label="log_return")
    axes[2].plot(zoom["end_date"], zoom[score_column], color="#8c564b", label="NLL score")
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1.0, label="threshold")
    axes[3].plot(zoom["end_date"], zoom["sigma_pred"], color="#ff7f0e", label="predicted sigma")
    axes[3].plot(zoom["end_date"], zoom["realized_vol_20"], color="#1f77b4", alpha=0.8, label="rolling vol 20")
    for ax in axes:
        _draw_weak_event_spans(ax, zoom, label_column)
        ax.axvline(center, color="red", alpha=0.4, linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
    axes[0].set_title(f"Figure 6 | {ticker} event zoom around {center.date()}")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    path = ticker_dir / "figure6_event_zoom.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    return paths


def run_nextday_report_figures(args) -> pd.DataFrame:
    outdir = Path(getattr(args, "out"))
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(getattr(args, "data_path", "datasets/SP500_event_taxonomy_w60"))
    score_column = str(getattr(args, "score_column", "score") or "score")
    label_column = str(getattr(args, "label_column", "endpoint_log_return_anomaly") or "endpoint_log_return_anomaly")
    threshold_quantile = float(getattr(args, "threshold_quantile", 0.99) or 0.99)
    student_csv = Path(getattr(args, "student_csv"))
    student_val_csv = Path(getattr(args, "student_validation_csv"))
    gaussian_csv = Path(getattr(args, "gaussian_csv", ""))
    metrics_csv = Path(getattr(args, "metrics_csv", ""))

    scores = pd.read_csv(student_csv, parse_dates=["end_date"])
    scores[score_column] = pd.to_numeric(scores[score_column], errors="coerce")
    scores[label_column] = pd.to_numeric(scores[label_column], errors="coerce").fillna(0).astype(int)
    threshold = _score_threshold(student_val_csv, threshold_quantile, score_column)

    requested_tickers = parse_csv_list(getattr(args, "tickers", None), [])
    if requested_tickers:
        tickers = requested_tickers
    else:
        tickers = (
            scores.groupby("ticker")[label_column]
            .sum()
            .sort_values(ascending=False)
            .head(int(getattr(args, "max_tickers", 6) or 6))
            .index.astype(str)
            .tolist()
        )

    figure_paths = []
    for ticker in tickers:
        figure_paths.extend(_plot_ticker_nextday_figures(scores, data_path, outdir, ticker, threshold, label_column, score_column))

    sample = scores.sample(min(len(scores), int(getattr(args, "scatter_sample", 25000) or 25000)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(sample[label_column].to_numpy() > 0, "#2ca02c", "#1f77b4")
    ax.scatter(sample["endpoint_log_return"].abs(), sample[score_column], c=colors, s=5, alpha=0.35)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label=f"val p99={threshold:.4g}")
    ax.set_xlabel("|endpoint log return|")
    ax.set_ylabel("Student-t NLL score")
    ax.set_title("Figure 3 | NLL vs absolute return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    scatter_path = outdir / "figures" / "figure3_nll_vs_abs_return_scatter.png"
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    figure_paths.append(str(scatter_path))

    events = _weak_event_intervals(scores, label_column)
    event_detection = _event_metrics(scores, events, threshold, score_column)
    events.to_csv(outdir / "event_table.csv", index=False)
    event_detection.to_csv(outdir / "event_detection.csv", index=False)

    top_anomalies = scores.sort_values(score_column, ascending=False).head(100)
    top_anomalies.to_csv(outdir / "top_anomalies.csv", index=False)

    model_rows = []
    for name, csv_path in [
        ("Student-t NLL Transformer", student_csv),
        ("Gaussian NLL Transformer", gaussian_csv),
    ]:
        if not csv_path or not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, parse_dates=["end_date"])
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
        frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce").fillna(0).astype(int)
        frame_events = _weak_event_intervals(frame, label_column)
        frame_detection = _event_metrics(frame, frame_events, float(frame[score_column].quantile(threshold_quantile)), score_column)
        model_rows.append(
            {
                "model": name,
                "precision_top1pct": _precision_at_top_fraction(frame, label_column, score_column, 0.01),
                "event_recall_top1pct": float(frame_detection["detected_at_threshold"].mean()) if not frame_detection.empty else np.nan,
                "hit_rate_top1pct": float(frame_detection["detected_at_threshold"].mean()) if not frame_detection.empty else np.nan,
                "mean_best_rank": float(frame_detection["best_rank"].replace(-1, np.nan).mean()) if not frame_detection.empty else np.nan,
            }
        )
    if "tail_z_abs" in scores.columns:
        z_frame = scores.copy()
        z_frame[score_column] = pd.to_numeric(z_frame["tail_z_abs"], errors="coerce")
        z_events = _weak_event_intervals(z_frame, label_column)
        z_detection = _event_metrics(z_frame, z_events, float(z_frame[score_column].quantile(threshold_quantile)), score_column)
        model_rows.append(
            {
                "model": "Rolling z-score proxy",
                "precision_top1pct": _precision_at_top_fraction(z_frame, label_column, score_column, 0.01),
                "event_recall_top1pct": float(z_detection["detected_at_threshold"].mean()) if not z_detection.empty else np.nan,
                "hit_rate_top1pct": float(z_detection["detected_at_threshold"].mean()) if not z_detection.empty else np.nan,
                "mean_best_rank": float(z_detection["best_rank"].replace(-1, np.nan).mean()) if not z_detection.empty else np.nan,
            }
        )
    metrics = pd.DataFrame(model_rows)
    if metrics_csv.exists():
        existing = pd.read_csv(metrics_csv)
        keep = [c for c in ["model", "weak_label_roc_auc", "weak_label_pr_auc"] if c in existing.columns]
        if keep:
            metrics = metrics.merge(existing[keep], on="model", how="left")
    metrics.to_csv(outdir / "metrics_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    bar_metrics = ["event_recall_top1pct", "precision_top1pct", "mean_best_rank", "weak_label_pr_auc"]
    titles = ["Event Recall@Top1%", "Precision@Top1%", "Mean Best Rank", "AUC-PR"]
    for ax, metric, title in zip(axes, bar_metrics, titles):
        if metric in metrics.columns:
            ax.bar(metrics["model"], pd.to_numeric(metrics[metric], errors="coerce"), color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(metrics)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Figure 7 | Model comparison")
    fig.tight_layout()
    bar_path = outdir / "figures" / "figure7_model_comparison_bar_chart.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    figure_paths.append(str(bar_path))

    report = outdir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Next-day NLL Visualization Report",
                "",
                f"Threshold: validation q{threshold_quantile:g} of `{score_column}` = `{threshold:.6g}`.",
                "Weak event labels use endpoint log-return 3-std flags from the exported score file.",
                "",
                "Generated figures:",
                *[f"- {path}" for path in figure_paths],
                "",
                "Skipped: reconstruction, association heatmap, and GARCH-t comparison are not plotted here because this fast evidence package did not run those models.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Done. Next-day report figures in {outdir}")
    return metrics

__all__ = [
    "AttentionPlotSummary",
    "ModelPanelSummary",
    "SpikeSummary",
    "add_final_score_columns",
    "add_verticals",
    "contiguous_intervals",
    "final_spike_mask",
    "interval_end",
    "load_price_frame",
    "mad_threshold",
    "parse_csv_list",
    "price_anomaly_starts",
    "plot_mad_ticker",
    "plot_attention_window",
    "plot_endpoint_logreturn_score_panel",
    "plot_model_diagnostic_panel",
    "robust_z",
    "run_attention_visualize",
    "run_mad_visualize",
    "run_dataset_logreturn_histograms",
    "run_endpoint_logreturn_score_panels",
    "run_model_diagnostic_panels",
    "run_nextday_report_figures",
]
