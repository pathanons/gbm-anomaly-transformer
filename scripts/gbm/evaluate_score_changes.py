#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.gbm.io import save_json


EVENT_COLORS = {
    "jump": "#2ca02c",
    "drop": "#9467bd",
    "volume_spike": "#17becf",
    "volatility_shock": "#8c564b",
    "regime_shift": "#e377c2",
    "is_anomaly": "#2ca02c",
}


def robust_scale(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    if clean.empty:
        return 0.0, 0.0
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    return median, 1.4826 * mad


def roc_auc_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    negatives = int((data["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = data["score"].rank(method="average")
    positive_rank_sum = float(ranks[data["y"] == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    if positives == 0:
        return float("nan")
    data = data.sort_values("score", ascending=False).reset_index(drop=True)
    true_positive_count = 0
    precision_sum = 0.0
    for idx, label in enumerate(data["y"], start=1):
        if int(label) == 1:
            true_positive_count += 1
            precision_sum += true_positive_count / idx
    return precision_sum / positives


def add_score_change_flags(
    scores: pd.DataFrame,
    rolling_window: int,
    min_periods: int,
    mad_k: float,
    fallback_quantile: float,
) -> pd.DataFrame:
    frames = []
    for ticker, frame in scores.groupby("ticker", sort=False):
        frame = frame.sort_values("end_date").reset_index(drop=True).copy()
        frame["score_delta"] = frame["score"].diff()
        frame["score_delta_abs"] = frame["score_delta"].abs()

        rolling = frame["score_delta_abs"].shift(1).rolling(rolling_window, min_periods=min_periods)
        rolling_median = rolling.median()
        rolling_mad = rolling.apply(lambda x: (x - x.median()).abs().median(), raw=False)
        frame["dynamic_threshold"] = rolling_median + mad_k * 1.4826 * rolling_mad

        fallback_threshold = float(frame["score_delta_abs"].dropna().quantile(fallback_quantile))
        frame["dynamic_threshold"] = frame["dynamic_threshold"].fillna(fallback_threshold)
        frame["score_change_pred"] = (frame["score_delta_abs"] > frame["dynamic_threshold"]).astype(int)
        frame["score_change_rank"] = frame["score_delta_abs"].rank(method="first", ascending=False)
        frame["threshold_source"] = f"rolling_median_plus_{mad_k:g}_mad"
        frame.loc[frame["score_delta_abs"].isna(), "score_change_pred"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metrics_from_dynamic_flags(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].astype(int)
    y_pred = frame["score_change_pred"].astype(int)
    score = frame["score_delta_abs"].fillna(0.0)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0

    if y_true.nunique() < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score_local(y_true, score))
        pr_auc = float(average_precision_score_local(y_true, score))

    return {
        "roc_auc_from_abs_score_change": roc_auc,
        "pr_auc_from_abs_score_change": pr_auc,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def event_columns_from_label_file(label_df: pd.DataFrame) -> list[str]:
    event_columns = [column for column in label_df.columns if column not in {"Date", "is_anomaly"}]
    if not event_columns and "is_anomaly" in label_df.columns:
        event_columns = ["is_anomaly"]
    return event_columns


def add_test_event_columns(results: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    frames = []
    all_event_columns = set()
    for ticker, frame in results.groupby("ticker", sort=False):
        frame = frame.copy()
        label_path = data_path / f"{ticker}_anomaly_label.csv"
        if not label_path.exists():
            frames.append(frame)
            continue

        label_df = pd.read_csv(label_path, parse_dates=["Date"])
        event_columns = event_columns_from_label_file(label_df)
        all_event_columns.update(event_columns)
        for event_column in event_columns:
            event_dates = label_df.loc[label_df[event_column].fillna(0).astype(float) > 0, "Date"]
            event_dates = sorted(pd.to_datetime(event_dates.drop_duplicates()))
            values = []
            for _, row in frame.iterrows():
                start_date = pd.to_datetime(row["start_date"])
                end_date = pd.to_datetime(row["end_date"])
                event_idx = bisect_left(event_dates, start_date)
                values.append(int(event_idx < len(event_dates) and event_dates[event_idx] <= end_date))
            frame[f"true_{event_column}"] = values
        frames.append(frame)

    annotated = pd.concat(frames, ignore_index=True)
    for event_column in sorted(all_event_columns):
        output_column = f"true_{event_column}"
        if output_column not in annotated.columns:
            annotated[output_column] = 0
        annotated[output_column] = annotated[output_column].fillna(0).astype(int)
    return annotated


def summarize_by_event_type(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_columns = [column for column in results.columns if column.startswith("true_") and column != "true_anomaly"]
    for column in sorted(event_columns):
        event_mask = results[column].fillna(0).astype(int) > 0
        flagged = results["score_change_pred"].astype(int) > 0
        total = int(event_mask.sum())
        caught = int((event_mask & flagged).sum())
        missed = int((event_mask & ~flagged).sum())
        rows.append(
            {
                "event_type": column.replace("true_", "", 1),
                "test_windows_with_event": total,
                "caught_by_score_change": caught,
                "missed_by_score_change": missed,
                "catch_rate": caught / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def next_date(dates: pd.Series, end_date: pd.Timestamp) -> pd.Timestamp:
    date_list = list(pd.to_datetime(dates))
    try:
        idx = date_list.index(pd.to_datetime(end_date))
    except ValueError:
        return pd.to_datetime(end_date)
    if idx + 1 < len(date_list):
        return date_list[idx + 1]
    return date_list[idx]


def contiguous_intervals(dates: pd.Series, mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = []
    start_idx = None
    date_list = list(pd.to_datetime(dates))
    mask_list = list(mask.astype(bool))

    for idx, flagged in enumerate(mask_list):
        if flagged and start_idx is None:
            start_idx = idx
        elif not flagged and start_idx is not None:
            intervals.append((date_list[start_idx], date_list[idx - 1]))
            start_idx = None

    if start_idx is not None:
        intervals.append((date_list[start_idx], date_list[len(mask_list) - 1]))
    return intervals


def load_test_window_event_dates(
    ticker: str,
    data_path: Path,
    frame: pd.DataFrame,
) -> dict[str, list[pd.Timestamp]]:
    label_path = data_path / f"{ticker}_anomaly_label.csv"
    if not label_path.exists():
        return {}

    label_df = pd.read_csv(label_path, parse_dates=["Date"])
    if label_df.empty:
        return {}

    event_columns = event_columns_from_label_file(label_df)

    test_windows = [(pd.to_datetime(row["start_date"]), pd.to_datetime(row["end_date"])) for _, row in frame.iterrows()]
    event_dates_by_type = {}
    for column in event_columns:
        test_event_dates = []
        event_dates = label_df.loc[label_df[column].fillna(0).astype(float) > 0, "Date"]
        if event_dates.empty:
            continue
        for event_date in event_dates.drop_duplicates().sort_values():
            event_date = pd.to_datetime(event_date)
            if any(start_date <= event_date <= end_date for start_date, end_date in test_windows):
                test_event_dates.append(event_date)
        if test_event_dates:
            event_dates_by_type[column] = test_event_dates
    return event_dates_by_type


def draw_event_labels(ax, event_dates_by_type: dict[str, list[pd.Timestamp]], alpha: float) -> None:
    for event_name, event_dates in event_dates_by_type.items():
        color = EVENT_COLORS.get(event_name, "#7f7f7f")
        label = f"True {event_name}"
        for idx, event_date in enumerate(event_dates):
            ax.axvline(pd.to_datetime(event_date), color=color, alpha=alpha, linewidth=0.9, label=label if idx == 0 else None, zorder=0)


def plot_ticker(
    ticker: str,
    frame: pd.DataFrame,
    data_path: Path,
    output_dir: Path,
    top_k: int,
    full_context: bool,
) -> None:
    ohlcv_path = data_path / f"{ticker}_ohlcv.csv"
    if not ohlcv_path.exists():
        print(f"[evaluate_score_changes] skip missing price file: {ohlcv_path}")
        return

    price_df = pd.read_csv(ohlcv_path, parse_dates=["Date"])
    frame = frame.sort_values("end_date").reset_index(drop=True)
    detection_start = pd.to_datetime(frame["start_date"].min())
    detection_end = pd.to_datetime(frame["end_date"].max())
    event_dates_by_type = load_test_window_event_dates(ticker, data_path, frame)
    if not full_context:
        price_df = price_df[(price_df["Date"] >= detection_start) & (price_df["Date"] <= detection_end)].copy()

    flagged = frame[frame["score_change_pred"] == 1].copy()
    top_flagged = flagged.sort_values("score_delta_abs", ascending=False).head(max(1, top_k))

    fig, (ax_price, ax_score, ax_delta) = plt.subplots(
        3,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.0]},
    )

    ax_price.plot(price_df["Date"], price_df["Close"], color="#111111", linewidth=1.3, label="Close")
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker} | score-change dynamic threshold anomaly chart")
    ax_price.grid(True, alpha=0.25)

    draw_event_labels(ax_price, event_dates_by_type, alpha=0.35)
    ax_price.legend(loc="upper left")

    ax_score.plot(frame["end_date"], frame["score"], color="#1f77b4", linewidth=1.0, label="Anomaly score")
    ax_score.scatter(
        flagged["end_date"],
        flagged["score"],
        color="#ff7f0e",
        s=22,
        label="Score-change anomaly",
        zorder=4,
    )
    ax_score.set_ylabel("Score")
    ax_score.grid(True, alpha=0.25)
    draw_event_labels(ax_score, event_dates_by_type, alpha=0.22)
    ax_score.legend(loc="upper left")

    ax_delta.plot(frame["end_date"], frame["score_delta_abs"], color="#4c78a8", linewidth=0.9, label="|score change|")
    ax_delta.plot(
        frame["end_date"],
        frame["dynamic_threshold"],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        label="Dynamic threshold",
    )
    ax_delta.scatter(
        flagged["end_date"],
        flagged["score_delta_abs"],
        color="#d62728",
        s=18,
        label="Flagged change",
        zorder=4,
    )
    ax_delta.set_ylabel("|Delta score|")
    ax_delta.set_xlabel("Date")
    ax_delta.grid(True, alpha=0.25)
    draw_event_labels(ax_delta, event_dates_by_type, alpha=0.22)
    ax_delta.legend(loc="upper left")

    for _, row in top_flagged.iterrows():
        start_date = pd.to_datetime(row["start_date"])
        end_date = pd.to_datetime(row["end_date"])
        span_end = next_date(price_df["Date"], end_date)
        ax_price.axvspan(start_date, span_end, color="#d62728", alpha=0.22)

    for _, row in flagged.iterrows():
        event_date = pd.to_datetime(row["end_date"])
        ax_price.axvline(event_date, color="#ff7f0e", alpha=0.35, linewidth=0.9)
        ax_score.axvline(event_date, color="#ff7f0e", alpha=0.18, linewidth=0.8)
        ax_delta.axvline(event_date, color="#ff7f0e", alpha=0.16, linewidth=0.8)

    if not full_context:
        ax_price.set_xlim(detection_start, detection_end)
        ax_score.set_xlim(detection_start, detection_end)
        ax_delta.set_xlim(detection_start, detection_end)

    fig.tight_layout()
    out_path = output_dir / f"{ticker}_score_change_dynamic_threshold_chart.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomalies from score changes without retraining the model")
    parser.add_argument("--source-exp-dir", default="results/old_experiments/experiment3_joint")
    parser.add_argument("--exp-name", default="experiment4_score_change_dynamic_threshold")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--scores-file", default=None)
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--mad-k", type=float, default=3.5)
    parser.add_argument("--fallback-quantile", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--skip-visualization", action="store_true")
    args = parser.parse_args()

    source_exp_dir = Path(args.source_exp_dir)
    scores_path = Path(args.scores_file) if args.scores_file else source_exp_dir / "reports" / "gbm_joint_test_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing test score file: {scores_path}")

    run_dir = Path("results") / "experiments" / args.exp_name
    reports_dir = run_dir / "reports"
    visual_dir = run_dir / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(scores_path, parse_dates=["start_date", "end_date"])
    if args.tickers:
        scores = scores[scores["ticker"].isin(args.tickers)].copy()
    if scores.empty:
        raise RuntimeError("No score rows available after filtering")

    results = add_score_change_flags(
        scores=scores,
        rolling_window=args.rolling_window,
        min_periods=args.min_periods,
        mad_k=args.mad_k,
        fallback_quantile=args.fallback_quantile,
    )
    results = add_test_event_columns(results, Path(args.data_path))

    metrics = metrics_from_dynamic_flags(results)
    metrics.update(
        {
            "source_scores": str(scores_path),
            "decision_rule": "score_change_pred = abs(score_t - score_t_minus_1) > dynamic_threshold_t",
            "dynamic_threshold": f"rolling median of prior abs score changes + {args.mad_k:g} * robust MAD",
            "rolling_window": args.rolling_window,
            "min_periods": args.min_periods,
            "fallback_quantile": args.fallback_quantile,
            "n_windows": int(len(results)),
            "ticker_count": int(results["ticker"].nunique()),
            "flagged_windows": int(results["score_change_pred"].sum()),
        }
    )

    scores_out = reports_dir / "score_change_dynamic_threshold_scores.csv"
    metrics_out = reports_dir / "score_change_dynamic_threshold_metrics.json"
    by_ticker_out = reports_dir / "score_change_dynamic_threshold_by_ticker.csv"
    by_event_out = reports_dir / "score_change_dynamic_threshold_by_event_type.csv"
    manifest_out = run_dir / "score_change_dynamic_threshold_manifest.json"

    results.to_csv(scores_out, index=False)
    save_json(metrics_out, metrics)
    summarize_by_event_type(results).to_csv(by_event_out, index=False)

    by_ticker = []
    for ticker, frame in results.groupby("ticker", sort=True):
        ticker_metrics = metrics_from_dynamic_flags(frame)
        by_ticker.append(
            {
                "ticker": ticker,
                "n_windows": int(len(frame)),
                "true_anomaly_windows": int(frame["y_true"].astype(int).sum()),
                "flagged_windows": int(frame["score_change_pred"].astype(int).sum()),
                "precision": ticker_metrics["precision"],
                "sensitivity": ticker_metrics["sensitivity"],
                "specificity": ticker_metrics["specificity"],
                "f1": ticker_metrics["f1_score"],
                "mean_abs_score_change": float(frame["score_delta_abs"].mean()),
                "max_abs_score_change": float(frame["score_delta_abs"].max()),
            }
        )
    pd.DataFrame(by_ticker).to_csv(by_ticker_out, index=False)

    manifest = {
        "experiment_name": args.exp_name,
        "source_experiment_dir": str(source_exp_dir),
        "source_scores": str(scores_path),
        "output_dir": str(run_dir),
        "reports": {
            "scores": str(scores_out),
            "metrics": str(metrics_out),
            "by_ticker": str(by_ticker_out),
            "by_event_type": str(by_event_out),
        },
        "visualizations": str(visual_dir),
        "decision_rule": metrics["decision_rule"],
        "parameters": {
            "rolling_window": args.rolling_window,
            "min_periods": args.min_periods,
            "mad_k": args.mad_k,
            "fallback_quantile": args.fallback_quantile,
            "top_k": args.top_k,
            "tickers": args.tickers,
        },
    }
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.skip_visualization:
        data_path = Path(args.data_path)
        for idx, (ticker, frame) in enumerate(results.groupby("ticker", sort=True), start=1):
            print(f"[evaluate_score_changes] plot {idx}/{results['ticker'].nunique()} {ticker}", flush=True)
            plot_ticker(ticker, frame, data_path, visual_dir, args.top_k, args.full_context)

    print(f"Saved dynamic score-change scores to {scores_out}")
    print(f"Saved metrics to {metrics_out}")
    print(f"Saved ticker summary to {by_ticker_out}")
    print(f"Saved manifest to {manifest_out}")


if __name__ == "__main__":
    main()
