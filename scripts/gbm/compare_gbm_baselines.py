#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.paths import get_run_dir


def load_gbm_scores(exp_name: str) -> pd.DataFrame:
    run_dir = get_run_dir(exp_name)
    path = run_dir / "reports" / "gbm_joint_test_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing GBM scores: {path}")
    frame = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    frame["model"] = "gbm"
    return frame


def load_baseline_scores(baseline_exp: str, baseline_name: str) -> pd.DataFrame:
    run_dir = get_run_dir(baseline_exp)
    path = run_dir / "reports" / f"{baseline_name}_test_scores.csv"
    if not path.exists():
        path = run_dir / "reports" / f"{baseline_name}_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline scores for {baseline_name}: {path}")
    frame = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    frame = frame.loc[frame["split"] == "test"] if "split" in frame.columns else frame
    frame["model"] = baseline_name
    return frame


def plot_ticker(gbm_df: pd.DataFrame, baseline_df: pd.DataFrame, ticker: str, output_path: Path, show_labels: bool) -> None:
    gbm_t = gbm_df.loc[gbm_df["ticker"] == ticker].sort_values("end_date")
    base_t = baseline_df.loc[baseline_df["ticker"] == ticker].sort_values("end_date")
    if gbm_t.empty or base_t.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_price, ax_score = axes

    dates = pd.to_datetime(gbm_t["end_date"])
    ax_score.plot(dates, gbm_t["score"], label="GBM canonical", color="#1f77b4", linewidth=1.2)
    base_dates = pd.to_datetime(base_t["end_date"])
    ax_score.plot(base_dates, base_t["score"], label=baseline_df["model"].iloc[0], color="#d62728", linewidth=1.0, alpha=0.85)
    ax_score.set_ylabel("Raw score")
    ax_score.legend(loc="upper left")
    ax_score.grid(True, alpha=0.3)

    if show_labels and "y_true" in gbm_t.columns:
        anomaly_mask = gbm_t["y_true"].astype(int) > 0
        for start, end in zip(gbm_t.loc[anomaly_mask, "start_date"], gbm_t.loc[anomaly_mask, "end_date"]):
            ax_score.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="#ff9896", alpha=0.15)

    ax_price.set_title(f"{ticker} | GBM vs {baseline_df['model'].iloc[0]}")
    ax_price.set_ylabel("Score context")
    ax_price.axis("off")

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GBM raw scores against baseline models per ticker")
    parser.add_argument("--gbm-exp-name", default="canonical_gbm_attention")
    parser.add_argument("--baseline-exp-name", default="baseline_comparison")
    parser.add_argument("--baseline", default="lstm_autoencoder")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-true-labels", action="store_true")
    args = parser.parse_args()

    gbm_df = load_gbm_scores(args.gbm_exp_name)
    baseline_df = load_baseline_scores(args.baseline_exp_name, args.baseline)

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = get_run_dir(args.baseline_exp_name) / "figures" / f"gbm_vs_{args.baseline}"

    tickers = sorted(set(gbm_df["ticker"]) & set(baseline_df["ticker"]))
    if args.ticker:
        tickers = [args.ticker]
    elif not args.all_tickers:
        tickers = tickers[:5]

    for ticker in tickers:
        out_path = output_root / f"gbm_vs_{args.baseline}_{ticker}.png"
        plot_ticker(gbm_df, baseline_df, ticker, out_path, args.show_true_labels)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
