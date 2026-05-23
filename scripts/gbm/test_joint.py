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
from src.gbm.score_dynamics import apply_score_change_threshold, apply_score_turning_point_rule


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the joint EXP3 GBM-aware model")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", default="experiment3_joint")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--recon-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
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

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_joint_loaders(
        data_path=args.data_path,
        tickers=args.tickers if args.tickers else discover_tickers(args.data_path),
        window_size=args.window_size,
        batch_size=args.batch_size,
        step=args.step,
        features=args.features,
        normalize_batch=args.normalize_batch,
        seed=args.seed,
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
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("[test_joint] scoring test set", flush=True)
    test_df = collect_joint_scores(model, test_loader, device, phase="test", recon_weight=args.recon_weight, association_weight=args.association_weight)
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    if threshold_score_column == "score_turning_point":
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
    metrics = binary_metrics(test_df["y_true"].tolist(), metric_scores.tolist(), metric_threshold)
    metrics.update(
        {
            "threshold": threshold,
            "threshold_score_column": threshold_score_column,
            "decision_rule": threshold_info.get("decision_rule", "score_gt_validation_quantile"),
            "n_windows": int(len(test_df)),
            "ticker_count": int(manifest["ticker"].nunique()),
            "anomaly_rate": float(test_df["y_true"].mean()),
            "mean_score": float(test_df["score"].mean()),
            "std_score": float(test_df["score"].std(ddof=0)),
            "mean_score_delta_abs": float(test_df.get("score_delta_abs", pd.Series(dtype=float)).fillna(0.0).mean()),
            "mean_score_curvature_abs": float(test_df.get("score_curvature_abs", pd.Series(dtype=float)).fillna(0.0).mean()),
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

    ticker_summary = (
        test_df.groupby("ticker")
        .apply(lambda frame: pd.Series(binary_metrics(frame["y_true"].tolist(), (frame[threshold_score_column].fillna(0.0) if threshold_score_column in frame else frame["score"]).tolist(), metric_threshold)))
        .reset_index()
    )
    ticker_summary.to_csv(reports_dir / "gbm_joint_metrics_by_ticker.csv", index=False)
    print(f"Saved scores to {scores_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved per-ticker scores to {by_ticker_dir}")


if __name__ == "__main__":
    main()
