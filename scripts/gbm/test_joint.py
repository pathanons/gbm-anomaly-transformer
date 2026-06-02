#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from src.gbm.model import AnomalyTransformer
from src.gbm.score_dynamics import add_test_event_columns, average_precision_score_local, roc_auc_score_local
from src.gbm.scoring import collect_joint_scores


def load_state_dict(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def score_summary(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
    score = frame["score"].astype(float)
    y_true = frame["y_true"].astype(int)
    output: dict[str, float | int] = {
        f"{prefix}n_windows": int(len(frame)),
        f"{prefix}anomaly_rate": float(y_true.mean()) if len(frame) else 0.0,
        f"{prefix}score_mean": float(score.mean()) if len(frame) else float("nan"),
        f"{prefix}score_std": float(score.std(ddof=0)) if len(frame) else float("nan"),
        f"{prefix}score_min": float(score.min()) if len(frame) else float("nan"),
        f"{prefix}score_max": float(score.max()) if len(frame) else float("nan"),
        f"{prefix}score_p50": float(score.quantile(0.50)) if len(frame) else float("nan"),
        f"{prefix}score_p90": float(score.quantile(0.90)) if len(frame) else float("nan"),
        f"{prefix}score_p95": float(score.quantile(0.95)) if len(frame) else float("nan"),
        f"{prefix}score_p99": float(score.quantile(0.99)) if len(frame) else float("nan"),
    }
    if y_true.nunique() >= 2:
        output[f"{prefix}roc_auc_from_score"] = float(roc_auc_score_local(y_true, score))
        output[f"{prefix}pr_auc_from_score"] = float(average_precision_score_local(y_true, score))
    else:
        output[f"{prefix}roc_auc_from_score"] = float("nan")
        output[f"{prefix}pr_auc_from_score"] = float("nan")
    return output


def summarize_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_columns = [column for column in frame.columns if column.startswith("true_")]
    for event_column in event_columns:
        event_mask = frame[event_column].fillna(0).astype(int) > 0
        normal_mask = ~event_mask
        row = {
            "event_type": event_column.replace("true_", ""),
            "event_windows": int(event_mask.sum()),
            "non_event_windows": int(normal_mask.sum()),
        }
        if event_mask.any():
            row.update(
                {
                    "event_score_mean": float(frame.loc[event_mask, "score"].mean()),
                    "event_score_p95": float(frame.loc[event_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"event_score_mean": float("nan"), "event_score_p95": float("nan")})
        if normal_mask.any():
            row.update(
                {
                    "non_event_score_mean": float(frame.loc[normal_mask, "score"].mean()),
                    "non_event_score_p95": float(frame.loc[normal_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"non_event_score_mean": float("nan"), "non_event_score_p95": float("nan")})
        rows.append(row)
    return pd.DataFrame(rows)


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
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    checkpoint_run_dir = get_run_dir(args.checkpoint_exp_name) if args.checkpoint_exp_name else run_dir
    checkpoint_path = checkpoint_run_dir / "models" / "gbm_joint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    print(f"[test_joint] loading checkpoint={checkpoint_path}")

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

    test_df = add_test_event_columns(test_df, Path(args.data_path))
    metrics = score_summary(test_df)
    metrics.update(
        {
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "ticker_count": int(manifest["ticker"].nunique()),
            "mean_reconstruction_error": float(test_df["reconstruction_error"].mean()),
            "mean_nll": float(test_df["nll"].mean()),
            "mean_divergence": float(test_df["divergence"].mean()),
            "mean_association_discrepancy": float(test_df["association_discrepancy"].mean()),
            "mean_tail_z_abs": float(test_df["tail_z_abs"].mean()),
        }
    )

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "gbm_joint_test_scores.csv"
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
