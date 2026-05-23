#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from src.gbm.data import build_joint_loaders, discover_tickers, get_run_dir, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.model import AnomalyTransformer
from src.gbm.scoring import collect_joint_scores


def load_state_dict(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the joint financial prior attention model")
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
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
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
    parser.add_argument("--threshold-method", default="quantile", choices=["quantile", "conformal"])
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[validate_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    checkpoint_path = run_dir / "models" / "gbm_joint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    print(f"[validate_joint] loading checkpoint={checkpoint_path}")

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
    print(f"[validate_joint] windows total={len(manifest)} | val={len(val_ds)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[validate_joint] val_batches={len(val_loader)} | test_preview_batches={len(test_loader)}")

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

    print("[validate_joint] scoring validation set", flush=True)
    val_df = collect_joint_scores(
        model,
        val_loader,
        device,
        phase="validate",
        dist_weight=args.dist_weight,
        recon_weight=args.recon_weight,
        divergence_weight=args.divergence_weight,
        association_weight=args.association_weight,
        predictive_distribution=args.predictive_distribution,
    )
    print("[validate_joint] scoring test preview set", flush=True)
    test_df = collect_joint_scores(
        model,
        test_loader,
        device,
        phase="test_preview",
        dist_weight=args.dist_weight,
        recon_weight=args.recon_weight,
        divergence_weight=args.divergence_weight,
        association_weight=args.association_weight,
        predictive_distribution=args.predictive_distribution,
    )
    if val_df.empty:
        raise RuntimeError("Validation scoring returned no rows")

    threshold_source = val_df.loc[val_df["y_true"] == 0, "score"]
    threshold_fit_subset = "validation_normal_windows"
    if threshold_source.empty:
        threshold_source = val_df["score"]
        threshold_fit_subset = "validation_all_windows"
    calibration_scores = threshold_source.to_numpy()
    threshold = float(np.quantile(calibration_scores, args.threshold_quantile))
    val_df["conformal_p_value"] = val_df["score"].apply(
        lambda score: (float((calibration_scores >= score).sum()) + 1.0) / (len(calibration_scores) + 1.0)
    )
    test_df["conformal_p_value"] = test_df["score"].apply(
        lambda score: (float((calibration_scores >= score).sum()) + 1.0) / (len(calibration_scores) + 1.0)
    )
    if args.threshold_method == "conformal":
        conformal_alpha = 1.0 - args.threshold_quantile
        val_df["y_pred"] = (val_df["conformal_p_value"] <= conformal_alpha).astype(int)
        test_df["y_pred"] = (test_df["conformal_p_value"] <= conformal_alpha).astype(int)
        threshold_score_column = "conformal_p_value"
        decision_rule = "conformal_p_value_lte_alpha"
    else:
        conformal_alpha = None
        val_df["y_pred"] = (val_df["score"] > threshold).astype(int)
        test_df["y_pred"] = (test_df["score"] > threshold).astype(int)
        threshold_score_column = "score"
        decision_rule = "score_gt_validation_normal_quantile"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = reports_dir / "gbm_joint_threshold.json"
    save_json(
        threshold_path,
        {
            "threshold_quantile": args.threshold_quantile,
            "threshold": threshold,
            "threshold_score_column": threshold_score_column,
            "decision_rule": decision_rule,
            "threshold_method": args.threshold_method,
            "conformal_alpha": conformal_alpha,
            "calibration_scores": [float(score) for score in calibration_scores.tolist()],
            "threshold_fit_subset": threshold_fit_subset,
            "dist_weight": args.dist_weight,
            "recon_weight": args.recon_weight,
            "divergence_weight": args.divergence_weight,
            "association_weight": args.association_weight,
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "val_score_mean": float(val_df["score"].mean()),
            "val_score_std": float(val_df["score"].std(ddof=0)),
            "val_windows": int(len(val_df)),
            "test_windows": int(len(test_df)),
            "ticker_count": int(manifest["ticker"].nunique()),
            "split_counts": manifest["split"].value_counts().to_dict(),
        },
    )

    val_df.to_csv(reports_dir / "gbm_joint_validation_scores.csv", index=False)
    test_df.to_csv(reports_dir / "gbm_joint_test_scores_preview.csv", index=False)
    print(f"Saved threshold to {threshold_path}")
    print(f"Validation windows: {len(val_df)} | Test windows: {len(test_df)}")


if __name__ == "__main__":
    main()
