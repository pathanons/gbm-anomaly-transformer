#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.paths import get_run_dir


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot joint EXP3 anomaly charts per ticker")
    parser.add_argument("--exp-name", default="experiment3_joint")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scores-file", default=None, help="Optional score CSV override; defaults to reports/gbm_joint_test_scores.csv")
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--tail-days", type=int, default=None, help="Plot only the latest N calendar days in the scored range")
    parser.add_argument("--spike-percentile", type=float, default=0.95)
    parser.add_argument(
        "--score-mav-windows",
        type=int,
        nargs="+",
        default=[20, 50],
        help="Moving-average window sizes applied to anomaly scores, not prices",
    )
    args = parser.parse_args()

    mav_windows = sorted(set(args.score_mav_windows))
    if any(window <= 0 for window in mav_windows):
        raise ValueError("--score-mav-windows values must be positive integers")

    run_dir = get_run_dir(args.exp_name)
    scores_path = Path(args.scores_file) if args.scores_file else run_dir / "reports" / "gbm_joint_test_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing joint test scores file: {scores_path}")

    scores_df = pd.read_csv(scores_path, parse_dates=["start_date", "end_date"])

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[visualize_joint] loading scores={scores_path}")
    print(f"[visualize_joint] tickers={scores_df['ticker'].nunique()} | rows={len(scores_df)} | output_dir={output_dir}")

    for ticker_idx, (ticker, score_df) in enumerate(scores_df.groupby("ticker"), start=1):
        print(f"[visualize_joint] ticker {ticker_idx}/{scores_df['ticker'].nunique()} -> {ticker} | windows={len(score_df)}", flush=True)
        ohlcv_path = Path(args.data_path) / f"{ticker}_ohlcv.csv"
        label_path = Path(args.data_path) / f"{ticker}_anomaly_label.csv"
        if not ohlcv_path.exists():
            print(f"[visualize_joint] skip missing price file: {ohlcv_path}")
            continue

        price_df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
        label_df = pd.read_csv(label_path, parse_dates=["Date"]) if label_path.exists() else None

        detection_start = score_df["start_date"].min()
        detection_end = score_df["end_date"].max()
        plot_start = detection_start
        plot_end = detection_end
        if args.tail_days is not None:
            plot_start = max(detection_start, detection_end - pd.Timedelta(days=max(1, args.tail_days)))

        score_df = score_df.sort_values("end_date").reset_index(drop=True)
        for window in mav_windows:
            score_df[f"score_mav{window}"] = score_df["score"].rolling(window=window, min_periods=1).mean()

        if args.full_context and args.tail_days is None:
            plot_start = price_df["Date"].min()
            plot_end = price_df["Date"].max()
        else:
            price_df = price_df[(price_df["Date"] >= plot_start) & (price_df["Date"] <= plot_end)].copy()

        score_df = score_df[(score_df["end_date"] >= plot_start) & (score_df["end_date"] <= plot_end)].reset_index(drop=True)
        if score_df.empty:
            print(f"[visualize_joint] skip {ticker}: no scored windows in selected plot range")
            continue

        fig, (ax_price, ax_score, ax_mav) = plt.subplots(
            3,
            1,
            figsize=(18, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2, 1.2]},
        )
        ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.3, label="Close")
        ax_price.set_ylabel("Close")
        ax_price.set_title(f"{ticker} | joint EXP3 raw price, score, and score MAV chart")
        ax_price.grid(True, alpha=0.25)

        ax_score.plot(score_df["end_date"], score_df["score"], color="#1f77b4", linewidth=1.0, label="Anomaly score")
        ax_score.set_ylabel("Score")
        ax_score.grid(True, alpha=0.25)
        ax_score.legend(loc="upper left")

        for window in mav_windows:
            ax_mav.plot(
                score_df["end_date"],
                score_df[f"score_mav{window}"],
                linewidth=1.1,
                label=f"MAV{window} of score",
            )
        ax_mav.set_ylabel("Score MAV")
        ax_mav.set_xlabel("Date")
        ax_mav.grid(True, alpha=0.25)
        ax_mav.legend(loc="upper left")

        if not args.full_context or args.tail_days is not None:
            ax_price.set_xlim(plot_start, plot_end)
            ax_score.set_xlim(plot_start, plot_end)
            ax_mav.set_xlim(plot_start, plot_end)

        if args.show_true_labels and label_df is not None:
            label_columns = [col for col in label_df.columns if col != "Date"]
            if label_columns:
                label_df = label_df[(label_df["Date"] >= plot_start) & (label_df["Date"] <= plot_end)].copy()
                true_mask = (label_df[label_columns].fillna(0) > 0).any(axis=1)
                true_intervals = contiguous_intervals(label_df["Date"], true_mask)
                for start_date, end_date in true_intervals:
                    span_end = interval_end(price_df["Date"], end_date)
                    ax_price.axvspan(start_date, span_end, color="#2ca02c", alpha=0.10)

        fig.tight_layout()
        out_path = output_dir / f"{ticker}_joint_anomaly_price_chart.png"
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
