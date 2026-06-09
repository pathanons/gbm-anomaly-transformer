#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.gbm.data import get_run_dir, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.runtime import build_gbm_model, build_runtime_loaders, checkpoint_path, load_state_dict, score_kwargs
from src.gbm.scoring import collect_joint_scores


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
    parser.add_argument("--output-root", default=None, help="Output root containing experiments/<exp-name>")
    parser.add_argument("--checkpoint-exp-name", default=None, help="Optional source experiment to load the checkpoint from")
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
    parser.add_argument("--dist-weight", "--nll-weight", dest="dist_weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
    parser.add_argument("--score-mode", default="legacy", choices=["legacy", "refactored", "qw2", "qw2_tail"])
    parser.add_argument("--quantile-count", type=int, default=21)
    parser.add_argument("--tail-weight-gamma", type=float, default=2.0)
    parser.add_argument("--tail-weight-power", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[validate_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name, args.output_root)
    model_path = checkpoint_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[validate_joint] loading checkpoint={model_path}")

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[validate_joint] windows total={len(manifest)} | val={len(val_ds)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[validate_joint] val_batches={len(val_loader)} | test_preview_batches={len(test_loader)}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    print("[validate_joint] scoring validation set", flush=True)
    val_df = collect_joint_scores(
        model,
        val_loader,
        device,
        phase="validate",
        **score_kwargs(args),
    )
    print("[validate_joint] scoring test preview set", flush=True)
    test_df = collect_joint_scores(
        model,
        test_loader,
        device,
        phase="test_preview",
        **score_kwargs(args),
    )
    if val_df.empty:
        raise RuntimeError("Validation scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        reports_dir / "gbm_joint_score_summary.json",
        {
            "dist_weight": args.dist_weight,
            "recon_weight": args.recon_weight,
            "divergence_weight": args.divergence_weight,
            "association_weight": args.association_weight,
            "score_mode": args.score_mode,
            "quantile_count": args.quantile_count,
            "tail_weight_gamma": args.tail_weight_gamma,
            "tail_weight_power": args.tail_weight_power,
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "val_score_mean": float(val_df["score"].mean()),
            "val_score_std": float(val_df["score"].std(ddof=0)),
            "val_score_min": float(val_df["score"].min()),
            "val_score_max": float(val_df["score"].max()),
            "test_preview_score_mean": float(test_df["score"].mean()),
            "test_preview_score_std": float(test_df["score"].std(ddof=0)),
            "test_preview_score_min": float(test_df["score"].min()),
            "test_preview_score_max": float(test_df["score"].max()),
            "val_legacy_score_mean": float(val_df["legacy_score"].mean()),
            "val_refactored_score_mean": float(val_df["refactored_score"].mean()),
            "val_qw2_score_mean": float(val_df["score_qw2"].mean()),
            "val_qw2_tail_score_mean": float(val_df["score_qw2_tail"].mean()),
            "test_preview_legacy_score_mean": float(test_df["legacy_score"].mean()),
            "test_preview_refactored_score_mean": float(test_df["refactored_score"].mean()),
            "test_preview_qw2_score_mean": float(test_df["score_qw2"].mean()),
            "test_preview_qw2_tail_score_mean": float(test_df["score_qw2_tail"].mean()),
            "val_windows": int(len(val_df)),
            "test_preview_windows": int(len(test_df)),
            "ticker_count": int(manifest["ticker"].nunique()),
            "split_counts": manifest["split"].value_counts().to_dict(),
        },
    )

    val_df.to_csv(reports_dir / "gbm_joint_validation_scores.csv", index=False)
    test_df.to_csv(reports_dir / "gbm_joint_test_scores_preview.csv", index=False)
    print(f"Saved raw score summary to {reports_dir / 'gbm_joint_score_summary.json'}")
    print(f"Validation windows: {len(val_df)} | Test preview windows: {len(test_df)}")


if __name__ == "__main__":
    main()
