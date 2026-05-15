#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.data import create_joint_manifest, discover_tickers, get_run_dir
from src.gbm.io import save_json
from src.gbm.metrics import binary_metrics


EPS = 1e-8


def rolling_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=0))


def mean_abs_return(returns: np.ndarray) -> float:
    return float(np.mean(np.abs(returns)))


def last_return_zscore(returns: np.ndarray) -> float:
    if len(returns) <= 1:
        return 0.0
    history = returns[:-1]
    return float(abs((returns[-1] - np.mean(history)) / (np.std(history, ddof=0) + EPS)))


def max_return_zscore(returns: np.ndarray) -> float:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    return float(np.max(np.abs((returns - mean) / (std + EPS))))


BASELINES: Dict[str, Callable[[np.ndarray], float]] = {
    "rolling_volatility": rolling_volatility,
    "mean_abs_return": mean_abs_return,
    "last_return_zscore": last_return_zscore,
    "max_return_zscore": max_return_zscore,
}


def score_manifest(
    manifest: pd.DataFrame,
    window_store: dict[str, dict[str, np.ndarray]],
    baseline_name: str,
    scorer: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        ticker_store = window_store[row.ticker]
        returns = ticker_store["returns"][int(row.start_idx) : int(row.start_idx) + int(row.end_idx - row.start_idx)]
        score = scorer(np.nan_to_num(returns.astype(float)))
        if not math.isfinite(score):
            score = 0.0
        rows.append(
            {
                "baseline": baseline_name,
                "sample_id": int(row.sample_id),
                "ticker": row.ticker,
                "split": row.split,
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "start_date": row.start_date,
                "end_date": row.end_date,
                "y_true": int(row.y_true),
                "score": float(score),
            }
        )
    return pd.DataFrame(rows)


def summarize_baseline(scores: pd.DataFrame, threshold_quantile: float) -> dict[str, float | int | str]:
    val_scores = scores.loc[scores["split"] == "val", "score"].to_numpy(dtype=float)
    test_scores = scores.loc[scores["split"] == "test"].copy()
    threshold = float(np.quantile(val_scores, threshold_quantile))
    metrics = binary_metrics(test_scores["y_true"].tolist(), test_scores["score"].tolist(), threshold)
    metrics.update(
        {
            "baseline": str(scores["baseline"].iloc[0]),
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
            "val_windows": int(len(val_scores)),
            "test_windows": int(len(test_scores)),
            "ticker_count": int(test_scores["ticker"].nunique()),
            "anomaly_rate": float(test_scores["y_true"].mean()),
            "mean_score": float(test_scores["score"].mean()),
            "std_score": float(test_scores["score"].std(ddof=0)),
        }
    )
    return metrics


def parse_baselines(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(BASELINES))
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown)}. Available: {', '.join(BASELINES)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware statistical anomaly baselines on GBM windows")
    parser.add_argument("--exp-name", default="statistical_baselines")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--baselines", default="rolling_volatility,mean_abs_return,last_return_zscore,max_return_zscore")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else discover_tickers(args.data_path)
    if not tickers:
        raise SystemExit(f"No tickers found in {args.data_path}")

    selected_baselines = parse_baselines(args.baselines)
    run_dir = get_run_dir(args.exp_name)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest, window_store, _ = create_joint_manifest(
        data_path=args.data_path,
        tickers=tickers,
        window_size=args.window_size,
        step=args.step,
        features=args.features,
        seed=args.seed,
    )

    summary_rows = []
    score_paths = {}
    for baseline_name in selected_baselines:
        print(f"[statistical_baselines] scoring {baseline_name}", flush=True)
        scores = score_manifest(manifest, window_store, baseline_name, BASELINES[baseline_name])
        metrics = summarize_baseline(scores, args.threshold_quantile)
        summary_rows.append(metrics)
        score_path = reports_dir / f"{baseline_name}_scores.csv"
        scores.to_csv(score_path, index=False)
        score_paths[baseline_name] = str(score_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "statistical_baselines_summary.csv"
    metrics_path = reports_dir / "statistical_baselines_metrics.json"
    manifest_path = run_dir / "statistical_baselines_manifest.json"
    summary.to_csv(summary_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    save_json(
        manifest_path,
        {
            "exp_name": args.exp_name,
            "data_path": args.data_path,
            "tickers": list(tickers),
            "seed": args.seed,
            "window_size": args.window_size,
            "step": args.step,
            "features": args.features,
            "threshold_quantile": args.threshold_quantile,
            "split_counts": manifest["split"].value_counts().to_dict(),
            "reports": {
                "summary": str(summary_path),
                "metrics": str(metrics_path),
                "scores": score_paths,
            },
            "selection_protocol": "threshold selected from validation scores only; test used for final scoring only",
        },
    )

    print(summary.to_string(index=False))
    print(f"Saved summary to {summary_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
