#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd

from src.gbm.data import get_run_dir, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.runtime import build_gbm_model, build_runtime_loaders, checkpoint_path, load_state_dict, score_kwargs
from src.gbm.score_dynamics import add_test_event_columns
from src.gbm.scoring import collect_joint_scores
from src.gbm.test import score_summary, summarize_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the joint financial prior attention model and save raw scores")
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
    parser.add_argument(
        "--fast-export",
        action="store_true",
        help="Write gbm_joint_test_scores.csv only, skipping event and by-ticker summaries",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name, args.output_root)
    model_path = checkpoint_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[test_joint] loading checkpoint={model_path}")

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[test_joint] windows total={len(manifest)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[test_joint] test_batches={len(test_loader)}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    print("[test_joint] scoring test set", flush=True)
    test_df = collect_joint_scores(
        model,
        test_loader,
        device,
        phase="test",
        **score_kwargs(args),
    )
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "gbm_joint_test_scores.csv"
    if args.fast_export:
        test_df.to_csv(scores_path, index=False)
        print(f"Saved raw test scores to {scores_path}")
        print("[test_joint] fast export complete; skipped event and by-ticker summaries")
        return

    test_df = add_test_event_columns(test_df, Path(args.data_path))
    metrics = score_summary(test_df)
    metrics.update(
        {
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "score_mode": args.score_mode,
            "quantile_count": args.quantile_count,
            "tail_weight_gamma": args.tail_weight_gamma,
            "tail_weight_power": args.tail_weight_power,
            "ticker_count": int(manifest["ticker"].nunique()),
            "mean_reconstruction_error": float(test_df["reconstruction_error"].mean()),
            "mean_nll": float(test_df["nll"].mean()),
            "mean_divergence": float(test_df["divergence"].mean()),
            "mean_refactored_divergence": float(test_df["refactored_divergence"].mean()),
            "mean_dist_qw2": float(test_df["dist_qw2"].mean()),
            "mean_dist_qw2_tail": float(test_df["dist_qw2_tail"].mean()),
            "mean_association_discrepancy": float(test_df["association_discrepancy"].mean()),
            "mean_tail_z_abs": float(test_df["tail_z_abs"].mean()),
            "mean_legacy_score": float(test_df["legacy_score"].mean()),
            "mean_refactored_score": float(test_df["refactored_score"].mean()),
            "mean_score_qw2": float(test_df["score_qw2"].mean()),
            "mean_score_qw2_tail": float(test_df["score_qw2_tail"].mean()),
        }
    )

    metrics_path = reports_dir / "gbm_joint_score_metrics.json"
    test_df.to_csv(scores_path, index=False)
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = []
    for ticker, group in test_df.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"gbm_joint_{ticker}_test_scores.csv", index=False)
        row = {"ticker": ticker}
        row.update(score_summary(group))
        ticker_rows.append(row)
    pd.DataFrame(ticker_rows).to_csv(reports_dir / "gbm_joint_score_metrics_by_ticker.csv", index=False)

    event_summary = summarize_events(test_df)
    if not event_summary.empty:
        event_summary.to_csv(reports_dir / "gbm_joint_score_summary_by_event_type.csv", index=False)

    print(f"Saved raw test scores to {scores_path}")
    print(f"Saved raw score metrics to {metrics_path}")


if __name__ == "__main__":
    main()
