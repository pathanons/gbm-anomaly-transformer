#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.io import save_json
from src.gbm.metrics import binary_metrics


def parse_quantiles(raw: str) -> list[float]:
    quantiles = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {value}")
        quantiles.append(value)
    if not quantiles:
        raise ValueError("At least one quantile is required")
    return quantiles


def load_gbm_scores(exp_name: str, reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    val_path = reports_dir / "gbm_joint_validation_scores.csv"
    test_path = reports_dir / "gbm_joint_test_scores.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation scores: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test scores: {test_path}")
    return pd.read_csv(val_path), pd.read_csv(test_path), exp_name


def load_split_scores(scores_file: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not scores_file.exists():
        raise FileNotFoundError(f"Missing scores file: {scores_file}")
    scores = pd.read_csv(scores_file)
    if "split" not in scores.columns:
        raise ValueError(f"{scores_file} must include a split column")
    val_scores = scores[scores["split"] == "val"].copy()
    test_scores = scores[scores["split"] == "test"].copy()
    if val_scores.empty or test_scores.empty:
        raise ValueError(f"{scores_file} must include non-empty val and test splits")
    return val_scores, test_scores, label


def require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {', '.join(missing)}")


def sweep_thresholds(
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    quantiles: list[float],
    score_column: str,
    label_column: str,
) -> pd.DataFrame:
    require_columns(val_scores, [score_column], "validation scores")
    require_columns(test_scores, [score_column, label_column], "test scores")

    val_score_values = val_scores[score_column].to_numpy(dtype=float)
    y_true = test_scores[label_column].astype(int).tolist()
    y_score = test_scores[score_column].astype(float).tolist()
    rows = []
    for quantile in quantiles:
        threshold = float(np.quantile(val_score_values, quantile))
        metrics = binary_metrics(y_true, y_score, threshold)
        y_pred = (np.asarray(y_score, dtype=float) > threshold).astype(int)
        rows.append(
            {
                "threshold_quantile": float(quantile),
                "threshold": threshold,
                **metrics,
                "predicted_positive_windows": int(y_pred.sum()),
                "predicted_positive_rate": float(y_pred.mean()),
                "test_windows": int(len(test_scores)),
                "test_anomaly_rate": float(np.mean(y_true)),
                "val_score_mean": float(np.mean(val_score_values)),
                "val_score_std": float(np.std(val_score_values, ddof=0)),
                "test_score_mean": float(np.mean(y_score)),
                "test_score_std": float(np.std(y_score, ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep validation-selected score thresholds on test scores")
    parser.add_argument("--exp-name", default=None, help="Experiment under results/experiments containing GBM reports")
    parser.add_argument("--scores-file", default=None, help="Alternative CSV with split, score, and label columns")
    parser.add_argument("--label", default=None, help="Label used in output filenames when --scores-file is used")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="y_true")
    parser.add_argument("--quantiles", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.975,0.99")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    quantiles = parse_quantiles(args.quantiles)
    if bool(args.exp_name) == bool(args.scores_file):
        raise SystemExit("Provide exactly one of --exp-name or --scores-file")

    if args.exp_name:
        reports_dir = Path("results") / "experiments" / args.exp_name / "reports"
        val_scores, test_scores, label = load_gbm_scores(args.exp_name, reports_dir)
        output_dir = Path(args.output_dir) if args.output_dir else reports_dir
    else:
        scores_file = Path(args.scores_file)
        label = args.label or scores_file.stem
        val_scores, test_scores, label = load_split_scores(scores_file, label)
        output_dir = Path(args.output_dir) if args.output_dir else scores_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = sweep_thresholds(
        val_scores=val_scores,
        test_scores=test_scores,
        quantiles=quantiles,
        score_column=args.score_column,
        label_column=args.label_column,
    )

    summary_path = output_dir / f"{label}_threshold_sweep.csv"
    manifest_path = output_dir / f"{label}_threshold_sweep_manifest.json"
    summary.to_csv(summary_path, index=False)
    save_json(
        manifest_path,
        {
            "label": label,
            "score_column": args.score_column,
            "label_column": args.label_column,
            "quantiles": quantiles,
            "val_windows": int(len(val_scores)),
            "test_windows": int(len(test_scores)),
            "selection_protocol": "each threshold is selected from validation scores only; metrics are evaluated on test scores",
            "summary": str(summary_path),
        },
    )

    print(summary.to_string(index=False))
    print(f"Saved threshold sweep to {summary_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
