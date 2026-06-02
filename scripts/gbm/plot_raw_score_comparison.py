#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.paths import get_run_dir
from src.gbm.score_variants import variant_label


def contiguous_intervals(dates, mask):
    intervals = []
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


def interval_end(dates, end_date):
    date_list = list(pd.to_datetime(dates))
    try:
        end_idx = date_list.index(pd.to_datetime(end_date))
    except ValueError:
        return pd.to_datetime(end_date)
    if end_idx + 1 < len(date_list):
        return date_list[end_idx + 1]
    return date_list[end_idx]


def load_series_manifest(manifest_arg: str) -> dict[str, Path]:
    series: dict[str, Path] = {}
    for entry in manifest_arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Expected name=path, got {entry!r}")
        name, path_str = entry.split("=", 1)
        path = Path(path_str.strip())
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Missing score file for {name.strip()}: {path}")
        series[name.strip()] = path
    if not series:
        raise ValueError("No score series provided")
    return series


def load_score_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    if "score" not in frame.columns:
        raise ValueError(f"Missing score column in {path}")
    return frame.sort_values("end_date").reset_index(drop=True)


def plot_ticker(
    ticker: str,
    price_df: pd.DataFrame,
    label_df: pd.DataFrame | None,
    series_frames: dict[str, pd.DataFrame],
    output_path: Path,
    show_true_labels: bool,
    title_prefix: str,
) -> None:
    filtered = {}
    date_start = None
    date_end = None
    for name, frame in series_frames.items():
        part = frame.loc[frame["ticker"] == ticker].sort_values("end_date").reset_index(drop=True)
        if part.empty:
            return
        filtered[name] = part
        date_start = part["end_date"].min() if date_start is None else min(date_start, part["end_date"].min())
        date_end = part["end_date"].max() if date_end is None else max(date_end, part["end_date"].max())

    price_df = price_df[(price_df["Date"] >= date_start) & (price_df["Date"] <= date_end)].copy()
    n_score_panels = len(filtered)
    fig, axes = plt.subplots(
        1 + n_score_panels,
        1,
        figsize=(18, 3.0 + 2.2 * n_score_panels),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1.2] * n_score_panels},
    )
    if n_score_panels == 0:
        return
    axes = list(np.atleast_1d(axes).ravel())

    ax_price = axes[0]
    ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.35, label="Close")
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{title_prefix}{ticker} | raw scores | {date_start.date()} to {date_end.date()}")
    ax_price.grid(True, alpha=0.25)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for panel_idx, (name, frame) in enumerate(filtered.items()):
        ax = axes[panel_idx + 1]
        color = colors[panel_idx % len(colors)]
        label = variant_label(name) if name in {
            "recon_only", "association_kl_only", "recon_plus_kl", "full_default",
            "nll_only", "divergence_only", "nll_plus_recon",
        } else name
        ax.plot(frame["end_date"], frame["score"], color=color, linewidth=1.05, label=label)
        ax.set_ylabel("Raw score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)

    if show_true_labels and label_df is not None:
        label_columns = [col for col in label_df.columns if col != "Date"]
        if label_columns:
            label_slice = label_df[(label_df["Date"] >= date_start) & (label_df["Date"] <= date_end)].copy()
            true_mask = (label_slice[label_columns].fillna(0) > 0).any(axis=1)
            true_intervals = contiguous_intervals(label_slice["Date"], true_mask)
            for start_date, end_date in true_intervals:
                span_end = interval_end(price_df["Date"], end_date)
                for ax in axes:
                    ax.axvspan(start_date, span_end, color="#2ca02c", alpha=0.10)

    for ax in axes:
        ax.set_xlim(date_start, date_end)
    axes[-1].set_xlabel("Date")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multiple raw score series per ticker (no thresholds)")
    parser.add_argument(
        "--series",
        required=True,
        help="Comma-separated name=path entries, e.g. recon=reports/recon_only_test_scores.csv,kl=...",
    )
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args()

    series_paths = load_series_manifest(args.series)
    series_frames = {name: load_score_frame(path) for name, path in series_paths.items()}

    tickers = set.intersection(*[set(frame["ticker"].astype(str).unique()) for frame in series_frames.values()])
    tickers = sorted(tickers)
    if args.ticker:
        tickers = [args.ticker]
    elif not args.all_tickers:
        tickers = tickers[:5]
    if args.limit is not None:
        tickers = tickers[: max(0, args.limit)]

    if not tickers:
        raise RuntimeError("No shared tickers across score files")

    output_dir = Path(args.output_dir) if args.output_dir else Path("figures/raw_score_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tickers and args.ticker is None:
        for idx, ticker in enumerate(tickers, start=1):
            out_png = output_dir / f"{ticker}_raw_score_comparison.png"
            if out_png.exists():
                print(f"[plot_raw_scores] {idx}/{len(tickers)} {ticker} skip (exists)", flush=True)
                continue
            command = [
                sys.executable,
                str(Path(__file__)),
                "--series",
                args.series,
                "--data-path",
                args.data_path,
                "--ticker",
                ticker,
                "--output-dir",
                str(output_dir),
            ]
            if args.show_true_labels:
                command.append("--show-true-labels")
            if args.title_prefix:
                command.extend(["--title-prefix", args.title_prefix])
            print(f"[plot_raw_scores] {idx}/{len(tickers)} {ticker}", flush=True)
            subprocess.run(command, check=True)
        return

    if not args.ticker:
        raise RuntimeError("Pass --ticker SYMBOL or --all-tickers")

    ohlcv_path = Path(args.data_path) / f"{args.ticker}_ohlcv.csv"
    label_path = Path(args.data_path) / f"{args.ticker}_anomaly_label.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing price file: {ohlcv_path}")

    price_df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    label_df = pd.read_csv(label_path, parse_dates=["Date"]) if label_path.exists() else None

    out_path = output_dir / f"{args.ticker}_raw_score_comparison.png"
    plot_ticker(
        args.ticker,
        price_df,
        label_df,
        series_frames,
        out_path,
        args.show_true_labels,
        args.title_prefix,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
