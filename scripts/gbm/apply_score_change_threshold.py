#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.gbm.paths import get_run_dir
from src.gbm.score_dynamics import apply_score_change_threshold, add_score_delta_columns
from src.gbm.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply score-change anomaly decisions to existing GBM score files")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite gbm_joint_test_scores.csv y_pred with score-change decisions")
    args = parser.parse_args()

    run_dir = get_run_dir(args.exp_name)
    reports_dir = run_dir / "reports"
    val_path = reports_dir / "gbm_joint_validation_scores.csv"
    test_path = reports_dir / "gbm_joint_test_scores.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation scores: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test scores: {test_path}")

    val_df = pd.read_csv(val_path, parse_dates=["start_date", "end_date"])
    test_df = pd.read_csv(test_path, parse_dates=["start_date", "end_date"])
    val_df = add_score_delta_columns(val_df)
    delta_values = val_df["score_delta_abs"].dropna().to_numpy()
    if len(delta_values) == 0:
        raise RuntimeError("No validation score deltas available for threshold selection")

    threshold = float(np.quantile(delta_values, args.threshold_quantile))
    val_out = apply_score_change_threshold(val_df, threshold)
    test_out = apply_score_change_threshold(test_df, threshold)

    val_out_path = reports_dir / "gbm_joint_validation_scores_score_change.csv"
    test_out_path = reports_dir / "gbm_joint_test_scores_score_change.csv"
    val_out.to_csv(val_out_path, index=False)
    test_out.to_csv(test_out_path, index=False)

    if args.overwrite:
        test_out.to_csv(test_path, index=False)
        val_out.to_csv(val_path, index=False)

    threshold_path = reports_dir / "gbm_joint_threshold_score_change.json"
    threshold_payload = {
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "threshold_score_column": "score_delta_abs",
        "decision_rule": "score_change_abs_gt_validation_quantile",
        "val_score_delta_abs_mean": float(val_out["score_delta_abs"].fillna(0.0).mean()),
        "val_score_delta_abs_std": float(val_out["score_delta_abs"].fillna(0.0).std(ddof=0)),
        "val_windows": int(len(val_out)),
        "test_windows": int(len(test_out)),
        "test_flagged_windows": int(test_out["y_pred"].sum()),
        "outputs": {
            "validation_scores": str(val_out_path),
            "test_scores": str(test_out_path),
            "overwrote_pipeline_scores": bool(args.overwrite),
        },
    }
    save_json(threshold_path, threshold_payload)

    if args.overwrite:
        save_json(reports_dir / "gbm_joint_threshold.json", threshold_payload)

    print(f"Saved score-change validation scores to {val_out_path}")
    print(f"Saved score-change test scores to {test_out_path}")
    print(f"Saved score-change threshold to {threshold_path}")
    print(f"threshold={threshold:.6f} | flagged_test_windows={int(test_out['y_pred'].sum())}/{len(test_out)}")
    if args.overwrite:
        print("Overwrote pipeline validation/test score files and threshold file")


if __name__ == "__main__":
    main()
