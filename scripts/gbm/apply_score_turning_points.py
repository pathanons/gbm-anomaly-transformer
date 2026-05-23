#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.gbm.paths import get_run_dir
from src.gbm.score_dynamics import apply_score_turning_point_rule
from src.gbm.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply score turning-point anomaly decisions to existing GBM score files")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite gbm_joint_test_scores.csv y_pred with turning-point decisions")
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
    val_out = apply_score_turning_point_rule(val_df)
    test_out = apply_score_turning_point_rule(test_df)

    val_out_path = reports_dir / "gbm_joint_validation_scores_score_turning.csv"
    test_out_path = reports_dir / "gbm_joint_test_scores_score_turning.csv"
    val_out.to_csv(val_out_path, index=False)
    test_out.to_csv(test_out_path, index=False)

    if args.overwrite:
        test_out.to_csv(test_path, index=False)
        val_out.to_csv(val_path, index=False)

    manifest_path = reports_dir / "gbm_joint_threshold_score_turning.json"
    payload = {
        "threshold": None,
        "threshold_score_column": "score_turning_point",
        "decision_rule": "score_turning_point_no_threshold",
        "description": "Flag a window when the score line changes direction: previous slope and next slope have opposite signs.",
        "val_windows": int(len(val_out)),
        "test_windows": int(len(test_out)),
        "test_flagged_windows": int(test_out["y_pred"].sum()),
        "outputs": {
            "validation_scores": str(val_out_path),
            "test_scores": str(test_out_path),
            "overwrote_pipeline_scores": bool(args.overwrite),
        },
    }
    save_json(manifest_path, payload)

    if args.overwrite:
        save_json(reports_dir / "gbm_joint_threshold.json", payload)

    print(f"Saved turning-point validation scores to {val_out_path}")
    print(f"Saved turning-point test scores to {test_out_path}")
    print(f"Saved turning-point manifest to {manifest_path}")
    print(f"flagged_test_windows={int(test_out['y_pred'].sum())}/{len(test_out)}")
    if args.overwrite:
        print("Overwrote pipeline validation/test score files and threshold file")


if __name__ == "__main__":
    main()
