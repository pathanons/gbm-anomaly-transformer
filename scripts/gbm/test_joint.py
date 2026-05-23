#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import torch

from src.gbm.data import build_joint_loaders, discover_tickers, get_run_dir, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.metrics import binary_metrics
from src.gbm.model import AnomalyTransformer
from src.gbm.scoring import collect_joint_scores
from src.gbm.score_dynamics import (
    add_test_event_columns,
    apply_score_change_threshold,
    apply_score_turning_point_rule,
    summarize_by_event_type,
    tolerance_metrics,
)


def load_state_dict(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the joint financial prior attention model")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--split-method", default="chronological", choices=["chronological", "random"])
    parser.add_argument("--purge-gap", type=int, default=None)
    parser.add_argument("--include-anomalous-train", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", default="experiment3_joint")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--predictive-distribution", default="gaussian", choices=["gaussian", "student_t"])
    parser.add_argument(
        "--association-mode",
        default="gaussian_log_return",
        choices=["gaussian_log_return", "canonical_gbm", "temporal", "none"],
    )
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
    parser.add_argument("--tolerance-windows", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    checkpoint_path = run_dir / "models" / "gbm_joint.pt"
    threshold_path = run_dir / "reports" / "gbm_joint_threshold.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing validation threshold file: {threshold_path}")

    print(f"[test_joint] loading checkpoint={checkpoint_path}")
    print(f"[test_joint] loading threshold={threshold_path}")

    with open(threshold_path, "r", encoding="utf-8") as handle:
        threshold_info = json.load(handle)
    threshold = float(threshold_info["threshold"])
    threshold_score_column = threshold_info.get("threshold_score_column", "score")
    threshold_method = threshold_info.get("threshold_method", "quantile")
    threshold_distribution = threshold_info.get("predictive_distribution")
    if threshold_distribution is not None and threshold_distribution != args.predictive_distribution:
        raise RuntimeError(
            f"Validation threshold was fit with predictive_distribution={threshold_distribution}, "
            f"but test requested {args.predictive_distribution}"
        )
    threshold_association_mode = threshold_info.get("association_mode")
    if threshold_association_mode is not None and threshold_association_mode != args.association_mode:
        raise RuntimeError(
            f"Validation threshold was fit with association_mode={threshold_association_mode}, "
            f"but test requested {args.association_mode}"
        )

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_joint_loaders(
        data_path=args.data_path,
        tickers=args.tickers if args.tickers else discover_tickers(args.data_path),
        window_size=args.window_size,
        batch_size=args.batch_size,
        step=args.step,
        features=args.features,
        normalize_batch=args.normalize_batch,
        seed=args.seed,
        split_method=args.split_method,
        purge_gap=args.purge_gap,
        train_normal_only=not args.include_anomalous_train,
    )
    print(f"[test_joint] windows total={len(manifest)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[test_joint] test_batches={len(test_loader)}")

    model = AnomalyTransformer(
        win_size=args.window_size,
        enc_in=input_dim,
        c_out=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        predictive_distribution=args.predictive_distribution,
        association_mode=args.association_mode,
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device))

    print("[test_joint] scoring test set", flush=True)
    test_df = collect_joint_scores(
        model,
        test_loader,
        device,
        phase="test",
        dist_weight=args.dist_weight,
        recon_weight=args.recon_weight,
        divergence_weight=args.divergence_weight,
        association_weight=args.association_weight,
        predictive_distribution=args.predictive_distribution,
    )
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    if threshold_score_column == "conformal_p_value":
        calibration_scores = threshold_info.get("calibration_scores")
        if not calibration_scores:
            raise RuntimeError("Missing conformal calibration scores")
        calibration = pd.Series(calibration_scores, dtype=float)
        test_df["conformal_p_value"] = test_df["score"].apply(
            lambda score: (float((calibration >= score).sum()) + 1.0) / (len(calibration) + 1.0)
        )
        conformal_alpha = float(threshold_info.get("conformal_alpha", 1.0 - threshold_info.get("threshold_quantile", 0.95)))
        test_df["y_pred"] = (test_df["conformal_p_value"] <= conformal_alpha).astype(int)
        metric_scores = -test_df["conformal_p_value"]
        metric_threshold = -conformal_alpha
    elif threshold_score_column == "score_turning_point":
        test_df = apply_score_turning_point_rule(test_df)
        metric_scores = test_df["score_turning_point"].fillna(0.0)
        metric_threshold = 0.5
    elif threshold_score_column in {"score_delta_abs", "score_curvature_abs"}:
        if threshold is None:
            raise RuntimeError(f"Missing threshold for {threshold_score_column}")
        test_df = apply_score_change_threshold(test_df, threshold, column=threshold_score_column)
        metric_scores = test_df[threshold_score_column].fillna(0.0)
        metric_threshold = threshold
    else:
        if threshold is None:
            raise RuntimeError("Missing score threshold")
        test_df["y_pred"] = (test_df["score"] > threshold).astype(int)
        metric_scores = test_df["score"]
        metric_threshold = threshold
    if "score_change_pred" not in test_df.columns:
        test_df["score_change_pred"] = test_df["y_pred"].astype(int)
    test_df = add_test_event_columns(test_df, Path(args.data_path))
    metrics = binary_metrics(test_df["y_true"].tolist(), metric_scores.tolist(), metric_threshold)
    metrics.update(tolerance_metrics(test_df, args.tolerance_windows))
    score_delta_abs = test_df["score_delta_abs"].fillna(0.0) if "score_delta_abs" in test_df else pd.Series([0.0])
    score_curvature_abs = test_df["score_curvature_abs"].fillna(0.0) if "score_curvature_abs" in test_df else pd.Series([0.0])
    metrics.update(
        {
            "threshold": threshold,
            "threshold_score_column": threshold_score_column,
            "decision_rule": threshold_info.get("decision_rule", "score_gt_validation_quantile"),
            "threshold_method": threshold_method,
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "n_windows": int(len(test_df)),
            "ticker_count": int(manifest["ticker"].nunique()),
            "anomaly_rate": float(test_df["y_true"].mean()),
            "mean_score": float(test_df["score"].mean()),
            "std_score": float(test_df["score"].std(ddof=0)),
            "mean_nll": float(test_df["nll"].mean()),
            "mean_tail_z_abs": float(test_df["tail_z_abs"].mean()),
            "tolerance_windows": int(args.tolerance_windows),
            "mean_score_delta_abs": float(score_delta_abs.mean()),
            "mean_score_curvature_abs": float(score_curvature_abs.mean()),
        }
    )

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "gbm_joint_test_scores.csv"
    metrics_path = reports_dir / "gbm_joint_metrics.json"
    test_df.to_csv(scores_path, index=False)
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in test_df.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"gbm_joint_{ticker}_test_scores.csv", index=False)

    ticker_rows = []
    for ticker, frame in test_df.groupby("ticker"):
        score_column = threshold_score_column if threshold_score_column in frame else "score"
        row = {"ticker": ticker}
        row.update(binary_metrics(frame["y_true"].tolist(), frame[score_column].fillna(0.0).tolist(), metric_threshold))
        ticker_rows.append(row)
    ticker_summary = pd.DataFrame(ticker_rows)
    ticker_summary.to_csv(reports_dir / "gbm_joint_metrics_by_ticker.csv", index=False)
    event_summary = summarize_by_event_type(test_df)
    event_summary.to_csv(reports_dir / "gbm_joint_metrics_by_event_type.csv", index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved per-ticker scores to {by_ticker_dir}")


if __name__ == "__main__":
    main()
