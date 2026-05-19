#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--spike-percentile", type=float, default=0.95)
    args = parser.parse_args()

    run_dir = get_run_dir(args.exp_name)
    scores_path = run_dir / "reports" / "gbm_joint_test_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing joint test scores file: {scores_path}")

    scores_df = pd.read_csv(scores_path, parse_dates=["start_date", "end_date"])
    threshold_path = run_dir / "reports" / "gbm_joint_threshold.json"
    threshold = None
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as handle:
            threshold = float(json.load(handle)["threshold"])

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
        if not args.full_context:
            price_df = price_df[(price_df["Date"] >= detection_start) & (price_df["Date"] <= detection_end)].copy()

        score_df = score_df.sort_values("end_date").reset_index(drop=True)
        score_diff = score_df["score"].diff()
        positive_diff = score_diff[score_diff > 0].dropna()
        if len(positive_diff) > 0:
            spike_threshold = float(positive_diff.quantile(args.spike_percentile))
        else:
            spike_threshold = float("inf")
        spike_mask = score_diff > spike_threshold
        spike_df = score_df.loc[score_df.index[spike_mask]].copy() if spike_mask.any() else score_df.iloc[0:0].copy()

        fig, (ax_price, ax_score) = plt.subplots(2, 1, figsize=(18, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.3, label="Close")
        ax_price.set_ylabel("Close")
        ax_price.set_title(f"{ticker} | joint EXP3 anomaly chart")
        ax_price.grid(True, alpha=0.25)

        ax_score.plot(score_df["end_date"], score_df["score"], color="#1f77b4", linewidth=1.0, label="Anomaly score")
        ax_score.scatter(
            score_df.loc[score_df["y_pred"] == 1, "end_date"],
            score_df.loc[score_df["y_pred"] == 1, "score"],
            color="#d62728",
            s=18,
            label="Predicted anomaly",
            zorder=3,
        )
        if threshold is not None:
            ax_score.axhline(threshold, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"Threshold {threshold:.4f}")
        ax_score.set_ylabel("Score")
        ax_score.set_xlabel("Date")
        ax_score.grid(True, alpha=0.25)
        ax_score.legend(loc="upper left")

        if not spike_df.empty:
            ax_score.scatter(spike_df["end_date"], spike_df["score"], color="#ff7f0e", s=24, label="Score spike", zorder=4)
            ax_score.legend(loc="upper left")

        highlighted = score_df[score_df["y_pred"].astype(int) > 0].copy()
        if not highlighted.empty:
            highlighted = highlighted.sort_values("score", ascending=False).head(max(1, args.top_k))
        else:
            highlighted = score_df.sort_values("score", ascending=False).head(max(1, args.top_k))

        for _, row in highlighted.iterrows():
            start_date = pd.to_datetime(row["start_date"])
            end_date = pd.to_datetime(row["end_date"])
            span_end = interval_end(price_df["Date"], end_date)
            ax_price.axvspan(start_date, span_end, color="#d62728", alpha=0.20)
            ax_score.axvline(pd.to_datetime(row["end_date"]), color="#d62728", alpha=0.20, linewidth=1.0)

        if not spike_df.empty:
            for _, row in spike_df.iterrows():
                spike_date = pd.to_datetime(row["end_date"])
                ax_price.axvline(spike_date, color="#ff7f0e", alpha=0.55, linewidth=1.1, linestyle="-")
                ax_score.axvline(spike_date, color="#ff7f0e", alpha=0.20, linewidth=0.9)

        if not args.full_context:
            ax_price.set_xlim(detection_start, detection_end)
            ax_score.set_xlim(detection_start, detection_end)

        if args.show_true_labels and label_df is not None:
            label_columns = [col for col in label_df.columns if col != "Date"]
            if label_columns:
                label_df = label_df[(label_df["Date"] >= detection_start) & (label_df["Date"] <= detection_end)].copy()
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
