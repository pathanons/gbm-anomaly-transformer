#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.gbm.paths import get_run_dir
from src.gbm.io import save_json
from src.gbm.score_dynamics import (
    add_score_change_flags,
    add_test_event_columns,
    metrics_from_dynamic_flags,
    summarize_by_event_type,
    tolerance_metrics,
)


def parse_number_list(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep score-change dynamic-threshold rules on fixed GBM test scores")
    parser.add_argument("--source-exp-dir", default="results/old_experiments/experiment3_joint")
    parser.add_argument("--scores-file", default=None)
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--exp-name", default="experiment4_score_change_rule_sweep")
    parser.add_argument("--rolling-windows", default="20,50,100")
    parser.add_argument("--mad-ks", default="1.5,2.0,2.5,3.0,3.5")
    parser.add_argument("--min-periods-ratio", type=float, default=0.4)
    parser.add_argument("--fallback-quantile", type=float, default=0.95)
    parser.add_argument("--tolerance-windows", default="0,3,5,10")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--save-best-scores", action="store_true")
    args = parser.parse_args()

    source_exp_dir = Path(args.source_exp_dir)
    scores_path = Path(args.scores_file) if args.scores_file else source_exp_dir / "reports" / "gbm_joint_test_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing score file: {scores_path}")

    run_dir = get_run_dir(args.exp_name)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rolling_windows = parse_number_list(args.rolling_windows, int)
    mad_ks = parse_number_list(args.mad_ks, float)
    tolerance_windows = parse_number_list(args.tolerance_windows, int)

    scores = pd.read_csv(scores_path, parse_dates=["start_date", "end_date"])
    if args.tickers:
        scores = scores[scores["ticker"].isin(args.tickers)].copy()
    if scores.empty:
        raise RuntimeError("No score rows available after filtering")

    print(f"[sweep_score_change_rules] source_scores={scores_path}")
    print(f"[sweep_score_change_rules] rows={len(scores)} | tickers={scores['ticker'].nunique()}")
    print(f"[sweep_score_change_rules] rolling_windows={rolling_windows} | mad_ks={mad_ks}")

    scores = add_test_event_columns(scores, Path(args.data_path))

    summary_rows = []
    event_rows = []
    best_frame = None
    best_f1 = -1.0

    config_count = len(rolling_windows) * len(mad_ks)
    config_idx = 0
    for rolling_window in rolling_windows:
        min_periods = max(1, int(round(rolling_window * args.min_periods_ratio)))
        for mad_k in mad_ks:
            config_idx += 1
            print(
                f"[sweep_score_change_rules] config {config_idx}/{config_count} "
                f"rolling_window={rolling_window} min_periods={min_periods} mad_k={mad_k}",
                flush=True,
            )
            result = add_score_change_flags(
                scores=scores,
                rolling_window=rolling_window,
                min_periods=min_periods,
                mad_k=mad_k,
                fallback_quantile=args.fallback_quantile,
            )
            metrics = metrics_from_dynamic_flags(result)
            row = {
                "rolling_window": rolling_window,
                "min_periods": min_periods,
                "mad_k": mad_k,
                "fallback_quantile": args.fallback_quantile,
                "n_windows": int(len(result)),
                "ticker_count": int(result["ticker"].nunique()),
                "flagged_windows": int(result["score_change_pred"].sum()),
                **metrics,
            }
            for tolerance_window in tolerance_windows:
                tol = tolerance_metrics(result, tolerance_window)
                prefix = f"tol{tolerance_window}"
                for key, value in tol.items():
                    if key == "tolerance_windows":
                        continue
                    row[f"{prefix}_{key.replace('tolerance_', '')}"] = value

            summary_rows.append(row)

            event_summary = summarize_by_event_type(result)
            event_summary.insert(0, "mad_k", mad_k)
            event_summary.insert(0, "min_periods", min_periods)
            event_summary.insert(0, "rolling_window", rolling_window)
            event_rows.append(event_summary)

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_frame = result

    summary = pd.DataFrame(summary_rows)
    event_summary = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

    summary_path = reports_dir / "score_change_rule_sweep_summary.csv"
    event_path = reports_dir / "score_change_rule_sweep_by_event_type.csv"
    best_path = reports_dir / "score_change_rule_sweep_best_configs.csv"
    manifest_path = run_dir / "score_change_rule_sweep_manifest.json"

    summary.to_csv(summary_path, index=False)
    event_summary.to_csv(event_path, index=False)

    best_configs = pd.concat(
        [
            summary.sort_values("f1_score", ascending=False).head(5).assign(selection="best_strict_f1"),
            summary.sort_values("sensitivity", ascending=False).head(5).assign(selection="best_strict_recall"),
            summary.sort_values("precision", ascending=False).head(5).assign(selection="best_strict_precision"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["rolling_window", "mad_k", "selection"])
    best_configs.to_csv(best_path, index=False)

    best_scores_path = None
    if args.save_best_scores and best_frame is not None:
        best_scores_path = reports_dir / "score_change_rule_sweep_best_scores.csv"
        best_frame.to_csv(best_scores_path, index=False)

    manifest = {
        "experiment_name": args.exp_name,
        "source_experiment_dir": str(source_exp_dir),
        "source_scores": str(scores_path),
        "output_dir": str(run_dir),
        "decision_rule": "score_change_pred = abs(score_t - score_t_minus_1) > rolling_median_prior_abs_delta + mad_k * robust_mad",
        "parameters": {
            "rolling_windows": rolling_windows,
            "mad_ks": mad_ks,
            "min_periods_ratio": args.min_periods_ratio,
            "fallback_quantile": args.fallback_quantile,
            "tolerance_windows": tolerance_windows,
            "tickers": args.tickers,
        },
        "reports": {
            "summary": str(summary_path),
            "by_event_type": str(event_path),
            "best_configs": str(best_path),
            "best_scores": str(best_scores_path) if best_scores_path else None,
        },
    }
    save_json(manifest_path, manifest)

    print(f"Saved sweep summary to {summary_path}")
    print(f"Saved event summary to {event_path}")
    print(f"Saved best configs to {best_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
