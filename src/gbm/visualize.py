"""Canonical visualization surface for GBM experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    delta_median: float
    delta_mad: float
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


def parse_csv_list(value: str | None, default: Iterable[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
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


def _date_tick_positions(dates: np.ndarray, count: int = 6) -> tuple[np.ndarray, list[str]]:
    length = len(dates)
    if length == 0:
        return np.asarray([], dtype=int), []
    positions = np.linspace(0, length - 1, num=min(count, length), dtype=int)
    labels = [str(pd.to_datetime(dates[pos]).date()) for pos in positions]
    return positions, labels


def plot_attention_window(
    manifest_row: pd.Series,
    outdir: Path,
    layer: int | str = 0,
    head: int | str = 0,
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
    endpoint_series = series[-1]
    endpoint_prior = prior[-1]
    endpoint_diff = np.abs(endpoint_series - endpoint_prior)

    outdir.mkdir(parents=True, exist_ok=True)
    ticker = str(manifest_row["ticker"])
    window_id = int(manifest_row["window_id"])
    outpath = outdir / f"{ticker}_window{window_id}_layer{layer_idx}_head{head_idx}.png"

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
        gridspec_kw={"height_ratios": [1.0, 0.65]},
    )
    ax_series, ax_prior, ax_diff = axes[0]
    ax_profile, ax_delta, ax_text = axes[1]
    vmax = max(float(series.max()), float(prior.max()), 1e-8)
    diff_vmax = max(float(diff_abs.max()), 1e-8)
    image_specs = [
        (ax_series, series, "Learned series attention", vmax, "viridis"),
        (ax_prior, prior, prior_title, vmax, "viridis"),
        (ax_diff, diff_abs, "|series - prior|", diff_vmax, "magma"),
    ]
    for ax, matrix, title, max_value, cmap in image_specs:
        im = ax.imshow(matrix, aspect="auto", origin="upper", vmin=0.0, vmax=max_value, cmap=cmap)
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
    ax_profile.grid(True, alpha=0.25)
    ax_profile.legend(loc="upper left")

    ax_delta.bar(x, endpoint_diff, color="#8c564b", width=0.9)
    ax_delta.set_title("Endpoint absolute difference")
    ax_delta.set_xlabel("Key position")
    ax_delta.set_ylabel("Abs diff")
    ax_delta.grid(True, axis="y", alpha=0.25)

    ax_text.axis("off")
    top_lag = int(manifest_row.get("endpoint_top_lag", len(endpoint_series) - 1 - int(np.argmax(endpoint_series))))
    top_weight = float(manifest_row.get("endpoint_top_weight", float(endpoint_series.max())))
    endpoint_l1 = float(manifest_row.get("endpoint_l1_mean", float(endpoint_diff.sum())))
    description = (
        f"Ticker: {ticker}\n"
        f"Window: {window_id}\n"
        f"Dates: {manifest_row.get('start_date', '')} to {manifest_row.get('end_date', '')}\n"
        f"Layer/head: {layer_idx}/{head_idx}\n"
        f"Association mode: {association_mode}\n"
        f"Score: {float(manifest_row.get('score', np.nan)):.6g}\n"
        f"Association discrepancy: {float(manifest_row.get('association_discrepancy', np.nan)):.6g}\n"
        f"Endpoint top lag: {top_lag}\n"
        f"Endpoint top weight: {top_weight:.6g}\n"
        f"Endpoint L1 gap: {endpoint_l1:.6g}\n\n"
        "Interpretation:\n"
        "Series is the transformer's learned attention.\n"
        "Prior is the configured GBM timestamp posterior.\n"
        "Large gaps mark departures from that prior."
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


def add_final_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"nll", "association_discrepancy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required score columns: {', '.join(missing)}")

    out = frame.copy()
    out["nll"] = pd.to_numeric(out["nll"], errors="coerce")
    out["association_discrepancy"] = pd.to_numeric(out["association_discrepancy"], errors="coerce")
    out["raw_sum"] = out["nll"].fillna(0) + out["association_discrepancy"].fillna(0)
    out["score_robust_z"] = robust_z(out["raw_sum"])
    out["delta_score_robust_z"] = out["score_robust_z"].diff().abs()
    return out


def final_spike_mask(frame: pd.DataFrame, k: float) -> tuple[pd.Series, MadThreshold]:
    stats = mad_threshold(frame["delta_score_robust_z"], k=k)
    return frame["delta_score_robust_z"] > stats.threshold, stats


def load_price_frame(ticker: str, price_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    path = price_dir / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return None

    price_df = pd.read_csv(path, parse_dates=["Date"])
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
) -> SpikeSummary | None:
    sub = scores_df[scores_df["ticker"].astype(str) == ticker].sort_values("end_date").reset_index(drop=True)
    if sub.empty:
        return None

    required_columns = {"end_date", "nll", "association_discrepancy"}
    missing = sorted(required_columns - set(sub.columns))
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {', '.join(missing)}")

    sub["end_date"] = pd.to_datetime(sub["end_date"])
    sub = add_final_score_columns(sub)

    spike_mask, threshold_stats = final_spike_mask(sub, mad_k)
    spike_dates = list(sub.loc[spike_mask, "end_date"])
    scored_dates = set(pd.to_datetime(sub["end_date"]))

    start = sub["end_date"].min()
    end = sub["end_date"].max()
    price_df = load_price_frame(ticker, price_dir, start, end)
    price_starts = price_anomaly_starts(price_df, scored_dates, price_z_thr)
    overlap_dates = sorted(set(spike_dates) & set(price_starts))

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{ticker}_mad_spikes.png"

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9, 0.9, 0.8, 0.7]},
    )
    ax_price, ax_raw, ax_z, ax_zsum, ax_dz = axes
    x = sub["end_date"]

    if price_df is not None:
        ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.0)
    else:
        ax_price.plot(x, np.full(len(sub), np.nan), color="#111111")
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)

    ax_raw.plot(x, sub["nll"], label="nll", color="#1f77b4")
    ax_raw.plot(x, sub["association_discrepancy"], label="association", color="#ff7f0e")
    ax_raw.plot(x, sub["raw_sum"], label="nll + association", color="#7f7f7f", linestyle=":")
    ax_raw.set_ylabel("Raw terms")
    ax_raw.legend(loc="upper left")
    ax_raw.grid(True, alpha=0.25)

    ax_z.plot(x, robust_z(sub["nll"]), label="nll robust-z", color="#9467bd")
    ax_z.plot(x, robust_z(sub["association_discrepancy"]), label="association robust-z", color="#17becf")
    ax_z.set_ylabel("Term z")
    ax_z.legend(loc="upper left")
    ax_z.grid(True, alpha=0.25)

    ax_zsum.plot(x, sub["score_robust_z"], label="robust-z(nll + association)", color="#8c564b")
    ax_zsum.set_ylabel("Score z")
    ax_zsum.legend(loc="upper left")
    ax_zsum.grid(True, alpha=0.25)

    ax_dz.plot(x, sub["delta_score_robust_z"], label="abs diff score z", color="#e377c2")
    ax_dz.axhline(
        threshold_stats.threshold,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"k={mad_k:g} MAD threshold={threshold_stats.threshold:.3g}",
    )
    ax_dz.set_ylabel("Abs diff")
    ax_dz.set_xlabel("Date")
    ax_dz.legend(loc="upper left")
    ax_dz.grid(True, alpha=0.25)

    add_verticals(axes, spike_dates, color="red", alpha=0.25, linewidth=0.9)
    for idx, label_column in enumerate(label_columns):
        if label_column not in sub.columns:
            continue
        mask = pd.to_numeric(sub[label_column], errors="coerce").fillna(0) > 0
        add_verticals(axes, list(sub.loc[mask, "end_date"]), color=plt.cm.tab10(idx % 10), alpha=0.6, linewidth=1.0)
    add_verticals(axes, price_starts, color="green", alpha=0.45, linewidth=1.2)
    add_verticals(axes, overlap_dates, color="lime", alpha=0.9, linewidth=1.6)

    fig.suptitle(f"{ticker} | model spikes={len(spike_dates)}, price_anoms={len(price_starts)}, overlaps={len(overlap_dates)}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return SpikeSummary(
        ticker=ticker,
        model_spikes=len(spike_dates),
        price_anoms=len(price_starts),
        overlaps=len(overlap_dates),
        threshold=threshold_stats.threshold,
        delta_median=threshold_stats.median,
        delta_mad=threshold_stats.mad,
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
    summary: list[SpikeSummary] = []
    for idx, ticker in enumerate(tickers, start=1):
        result = plot_mad_ticker(scores_df, ticker, outdir, label_columns, price_dir, price_z_thr, mad_k=mad_k)
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
    summaries = []
    for _, row in manifest.iterrows():
        summaries.append(plot_attention_window(row, outdir, layer=layer, head=head))
    summary_df = pd.DataFrame([summary.__dict__ for summary in summaries])
    summary_path = outdir / "attention_plot_summary.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Done. Attention plots in {outdir} summary: {summary_path}")
    return summary_df

__all__ = [
    "AttentionPlotSummary",
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
    "robust_z",
    "run_attention_visualize",
    "run_mad_visualize",
]
