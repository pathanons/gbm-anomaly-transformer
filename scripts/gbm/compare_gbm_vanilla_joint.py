#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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


def score_strength(frame: pd.DataFrame) -> pd.Series:
    std = frame["score"].std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=frame.index)
    return (frame["score"] - frame["score"].mean()) / std


def choose_focus_range(
    gbm_df: pd.DataFrame,
    vanilla_df: pd.DataFrame,
    view: str,
    focus_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    detection_start = min(gbm_df["start_date"].min(), vanilla_df["start_date"].min())
    detection_end = max(gbm_df["end_date"].max(), vanilla_df["end_date"].max())
    if view == "full":
        return detection_start, detection_end, "full test range"

    focus_delta = pd.Timedelta(days=max(30, focus_days))
    if view == "tail":
        return max(detection_start, detection_end - focus_delta), detection_end, "latest test windows"

    candidates = []
    for model_name, frame in [("GBM", gbm_df), ("Vanilla", vanilla_df)]:
        work = frame.copy()
        work["strength"] = score_strength(work)
        work["priority"] = 0
        work.loc[work["y_true"].astype(int) > 0, "priority"] = 1
        work.loc[work["y_pred"].astype(int) > 0, "priority"] = 2
        work.loc[(work["y_true"].astype(int) > 0) & (work["y_pred"].astype(int) > 0), "priority"] = 3
        work = work[work["priority"] > 0]
        if work.empty:
            continue
        work["model"] = model_name
        candidates.append(work[["start_date", "end_date", "y_true", "y_pred", "score", "strength", "priority", "model"]])

    if not candidates:
        return max(detection_start, detection_end - focus_delta), detection_end, "latest test windows"

    candidate_df = pd.concat(candidates, ignore_index=True)
    best = candidate_df.sort_values(["priority", "strength", "end_date"], ascending=[False, False, False]).iloc[0]
    center = pd.to_datetime(best["end_date"])
    focus_start = max(detection_start, center - focus_delta / 2)
    focus_end = min(detection_end, center + focus_delta / 2)
    if focus_end - focus_start < focus_delta:
        if focus_start == detection_start:
            focus_end = min(detection_end, focus_start + focus_delta)
        elif focus_end == detection_end:
            focus_start = max(detection_start, focus_end - focus_delta)
    reason = f"{best['model']} high-confidence test anomaly near {center.date()}"
    return focus_start, focus_end, reason


def load_scores(exp_name: str) -> tuple[pd.DataFrame, float | None]:
    run_dir = get_run_dir(exp_name)
    scores_path = run_dir / "reports" / "gbm_joint_test_scores.csv"
    if not scores_path.exists():
        scores_path = run_dir / "reports" / "vanilla_joint_test_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores file for {exp_name}: {scores_path}")
    scores_df = pd.read_csv(scores_path, parse_dates=["start_date", "end_date"])

    threshold = None
    for threshold_name in ["gbm_joint_threshold.json", "vanilla_joint_threshold.json"]:
        threshold_path = run_dir / "reports" / threshold_name
        if threshold_path.exists():
            with open(threshold_path, "r", encoding="utf-8") as handle:
                threshold = float(json.load(handle)["threshold"])
            break
    return scores_df, threshold


def load_metrics(exp_name: str) -> dict[str, object]:
    run_dir = get_run_dir(exp_name)
    for metrics_name in ["gbm_joint_metrics.json", "vanilla_joint_metrics.json"]:
        metrics_path = run_dir / "reports" / metrics_name
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GBM and vanilla anomaly scores for one ticker")
    parser.add_argument("--gbm-exp-name", required=True)
    parser.add_argument("--vanilla-exp-name", required=True)
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--ticker", default=None, help="Ticker to plot, or ALL to plot every ticker shared by both runs")
    parser.add_argument("--all-tickers", action="store_true", help="Plot every ticker shared by both runs")
    parser.add_argument("--limit", type=int, default=None, help="Optional max tickers to plot when using --all-tickers/--ticker ALL")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--hide-true-labels", action="store_true")
    parser.add_argument(
        "--view",
        choices=["anomaly", "tail", "full"],
        default="anomaly",
        help="Which date range to plot: strongest test anomaly, latest test windows, or full test range",
    )
    parser.add_argument("--focus-days", type=int, default=540, help="Approximate days shown for anomaly/tail views")
    args = parser.parse_args()

    gbm_df, gbm_threshold = load_scores(args.gbm_exp_name)
    vanilla_df, vanilla_threshold = load_scores(args.vanilla_exp_name)

    run_all = args.all_tickers or (args.ticker is not None and args.ticker.upper() == "ALL")
    if run_all:
        gbm_tickers = set(gbm_df["ticker"].dropna().astype(str).unique())
        vanilla_tickers = set(vanilla_df["ticker"].dropna().astype(str).unique())
        tickers = sorted(gbm_tickers & vanilla_tickers)
        if args.limit is not None:
            tickers = tickers[: max(0, args.limit)]
        if not tickers:
            raise RuntimeError("No shared tickers found between GBM and vanilla score files")

        print(f"[compare_gbm_vanilla_joint] plotting {len(tickers)} tickers")
        for idx, ticker in enumerate(tickers, start=1):
            command = [
                sys.executable,
                str(Path(__file__)),
                "--gbm-exp-name",
                args.gbm_exp_name,
                "--vanilla-exp-name",
                args.vanilla_exp_name,
                "--data-path",
                args.data_path,
                "--ticker",
                ticker,
                "--view",
                args.view,
                "--focus-days",
                str(args.focus_days),
            ]
            if args.output_dir:
                command.extend(["--output-dir", args.output_dir])
            if args.show_true_labels:
                command.append("--show-true-labels")
            if args.hide_true_labels:
                command.append("--hide-true-labels")

            print(f"[compare_gbm_vanilla_joint] {idx}/{len(tickers)} {ticker}", flush=True)
            subprocess.run(command, check=True)
        return

    if not args.ticker:
        raise RuntimeError("Please pass --ticker AAPL, --ticker ALL, or --all-tickers")

    gbm_metrics = load_metrics(args.gbm_exp_name)
    vanilla_metrics = load_metrics(args.vanilla_exp_name)

    gbm_df = gbm_df[gbm_df["ticker"] == args.ticker].sort_values("end_date").reset_index(drop=True)
    vanilla_df = vanilla_df[vanilla_df["ticker"] == args.ticker].sort_values("end_date").reset_index(drop=True)
    if gbm_df.empty:
        raise RuntimeError(f"No GBM score rows found for ticker {args.ticker}")
    if vanilla_df.empty:
        raise RuntimeError(f"No vanilla score rows found for ticker {args.ticker}")

    ohlcv_path = Path(args.data_path) / f"{args.ticker}_ohlcv.csv"
    label_path = Path(args.data_path) / f"{args.ticker}_anomaly_label.csv"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"Missing price file: {ohlcv_path}")

    price_df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    label_df = pd.read_csv(label_path, parse_dates=["Date"]) if label_path.exists() else None

    detection_start, detection_end, focus_reason = choose_focus_range(gbm_df, vanilla_df, args.view, args.focus_days)
    price_df = price_df[(price_df["Date"] >= detection_start) & (price_df["Date"] <= detection_end)].copy()
    gbm_plot_df = gbm_df[(gbm_df["end_date"] >= detection_start) & (gbm_df["end_date"] <= detection_end)].copy()
    vanilla_plot_df = vanilla_df[(vanilla_df["end_date"] >= detection_start) & (vanilla_df["end_date"] <= detection_end)].copy()
    if gbm_plot_df.empty or vanilla_plot_df.empty:
        raise RuntimeError(f"No score rows found for selected view={args.view} between {detection_start.date()} and {detection_end.date()}")

    output_dir = Path(args.output_dir) if args.output_dir else get_run_dir(f"{args.gbm_exp_name}_vs_{args.vanilla_exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for model_name, metrics in [("gbm", gbm_metrics), ("vanilla", vanilla_metrics)]:
        summary_rows.append(
            {
                "model": model_name,
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "threshold": metrics.get("threshold"),
                "mean_score": metrics.get("mean_score"),
                "std_score": metrics.get("std_score"),
                "n_windows": metrics.get("n_windows"),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.ticker}_comparison_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved comparison metrics to {summary_path}")

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    ax_price, ax_gbm, ax_vanilla = axes

    ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.35, label="Close")
    ax_price.set_ylabel("Close")
    ax_price.set_title(
        f"{args.ticker} | GBM vs Vanilla anomaly scores | {detection_start.date()} to {detection_end.date()} ({focus_reason})"
    )
    ax_price.grid(True, alpha=0.25)

    ax_gbm.plot(gbm_plot_df["end_date"], gbm_plot_df["score"], color="#1f77b4", linewidth=1.05, label="GBM score")
    ax_gbm.scatter(
        gbm_plot_df.loc[gbm_plot_df["y_pred"] == 1, "end_date"],
        gbm_plot_df.loc[gbm_plot_df["y_pred"] == 1, "score"],
        color="#d62728",
        s=24,
        label="Predicted anomaly",
        zorder=3,
    )
    if gbm_threshold is not None:
        ax_gbm.axhline(gbm_threshold, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"GBM threshold {gbm_threshold:.4f}")
    ax_gbm.set_ylabel("GBM score")
    ax_gbm.grid(True, alpha=0.25)
    ax_gbm.legend(loc="upper left")

    ax_vanilla.plot(vanilla_plot_df["end_date"], vanilla_plot_df["score"], color="#2ca02c", linewidth=1.05, label="Vanilla score")
    ax_vanilla.scatter(
        vanilla_plot_df.loc[vanilla_plot_df["y_pred"] == 1, "end_date"],
        vanilla_plot_df.loc[vanilla_plot_df["y_pred"] == 1, "score"],
        color="#d62728",
        s=24,
        label="Predicted anomaly",
        zorder=3,
    )
    if vanilla_threshold is not None:
        ax_vanilla.axhline(vanilla_threshold, color="#ff7f0e", linestyle="--", linewidth=1.1, label=f"Vanilla threshold {vanilla_threshold:.4f}")
    ax_vanilla.set_ylabel("Vanilla score")
    ax_vanilla.set_xlabel("Date")
    ax_vanilla.grid(True, alpha=0.25)
    ax_vanilla.legend(loc="upper left")

    if (args.show_true_labels or not args.hide_true_labels) and label_df is not None:
        label_columns = [col for col in label_df.columns if col != "Date"]
        if label_columns:
            label_df = label_df[(label_df["Date"] >= detection_start) & (label_df["Date"] <= detection_end)].copy()
            true_mask = (label_df[label_columns].fillna(0) > 0).any(axis=1)
            true_intervals = contiguous_intervals(label_df["Date"], true_mask)
            for start_date, end_date in true_intervals:
                span_end = interval_end(price_df["Date"], end_date)
                for ax in axes:
                    ax.axvspan(start_date, span_end, color="#2ca02c", alpha=0.10)

    for ax in axes:
        ax.set_xlim(detection_start, detection_end)

    fig.tight_layout()
    out_path = output_dir / f"{args.ticker}_gbm_vs_vanilla_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
