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


def parse_csv_list(value: str | None, default: Iterable[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


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

__all__ = [
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
    "robust_z",
    "run_mad_visualize",
]
