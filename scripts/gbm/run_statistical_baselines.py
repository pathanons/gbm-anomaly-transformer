#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.baselines.statistical import BASELINE_NAMES, STATISTICAL_BASELINES, score_manifest
from src.gbm.data import create_joint_manifest, discover_tickers, get_run_dir
from src.gbm.io import save_json
from src.gbm.metrics import binary_metrics


def summarize_baseline(scores: pd.DataFrame, threshold_quantile: float) -> dict[str, float | int | str]:
    val_scores = scores.loc[scores["split"] == "val", "score"].to_numpy(dtype=float)
    test_scores = scores.loc[scores["split"] == "test"].copy()
    threshold = float(pd.Series(val_scores).quantile(threshold_quantile))
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
    unknown = sorted(set(names) - set(STATISTICAL_BASELINES))
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown)}. Available: {', '.join(BASELINE_NAMES)}")
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
    parser.add_argument(
        "--baselines",
        default=",".join(BASELINE_NAMES[:4]),
        help=f"Comma-separated names. Available: {', '.join(BASELINE_NAMES)}",
    )
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
        scores = score_manifest(manifest, window_store, baseline_name, STATISTICAL_BASELINES[baseline_name])
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
