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
from src.gbm.vanilla_anomaly_transformer import VanillaAnomalyTransformer
from scripts.gbm.validate_vanilla_joint import collect_scores


def score_summary(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
    score = frame["score"].astype(float)
    y_true = frame["y_true"].astype(int)
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the vanilla Anomaly Transformer and save raw scores")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", default="experiment3_vanilla_joint")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--prior-type", default="gaussian", choices=["gaussian", "powerlaw"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test_vanilla_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    checkpoint_path = run_dir / "models" / "vanilla_joint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    print(f"[test_vanilla_joint] loading checkpoint={checkpoint_path}")

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
    print(f"[test_vanilla_joint] windows total={len(manifest)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[test_vanilla_joint] test_batches={len(test_loader)}")

    model = VanillaAnomalyTransformer(
        win_size=args.window_size,
        enc_in=input_dim,
        c_out=input_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        prior_type=args.prior_type,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("[test_vanilla_joint] scoring test set", flush=True)
    test_df = collect_scores(model, test_loader, device, phase="test", temperature=args.temperature)
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    metrics = score_summary(test_df)
    metrics.update(
        {
            "prior_type": args.prior_type,
            "temperature": args.temperature,
            "ticker_count": int(manifest["ticker"].nunique()),
            "mean_reconstruction_error": float(test_df["reconstruction_error"].mean()),
            "mean_series_loss": float(test_df["series_loss"].mean()),
            "mean_prior_loss": float(test_df["prior_loss"].mean()),
        }
    )

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "vanilla_joint_test_scores.csv"
    metrics_path = reports_dir / "vanilla_joint_score_metrics.json"
    test_df.to_csv(scores_path, index=False)
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = []
    for ticker, group in test_df.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"vanilla_joint_{ticker}_test_scores.csv", index=False)
        row = {"ticker": ticker}
        row.update(score_summary(group))
        ticker_rows.append(row)
    pd.DataFrame(ticker_rows).to_csv(reports_dir / "vanilla_joint_score_metrics_by_ticker.csv", index=False)

    print(f"Saved raw test scores to {scores_path}")
    print(f"Saved raw score metrics to {metrics_path}")


if __name__ == "__main__":
    main()
