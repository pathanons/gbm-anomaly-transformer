#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from src.gbm.data import build_joint_loaders, discover_tickers, get_run_dir, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.vanilla_anomaly_transformer import VanillaAnomalyTransformer
from scripts.gbm.train_vanilla_joint import association_losses


def collect_scores(model, loader, device, phase: str, temperature: float = 50.0):
    import pandas as pd
    import torch.nn as nn

    model.eval()
    rows = []
    criterion = nn.MSELoss(reduction="none")
    batch_total = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            recon, series_list, prior_list, _ = model(x)
            recon_error = criterion(recon, x).mean(dim=-1)
            series_loss, prior_loss = association_losses(series_list, prior_list, temperature)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            score = (metric * recon_error).mean(dim=-1)

            if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                print(f"    [{phase}] batch {batch_idx}/{batch_total} | rows={len(score)}", flush=True)

            meta = batch["meta"]
            for idx in range(len(score)):
                rows.append(
                    {
                        "ticker": meta["ticker"][idx],
                        "split": meta["split"][idx],
                        "window_id": int(meta["window_id"][idx]),
                        "start_idx": int(meta["start_idx"][idx]),
                        "end_idx": int(meta["end_idx"][idx]),
                        "start_date": meta["start_date"][idx],
                        "end_date": meta["end_date"][idx],
                        "y_true": int(batch["y"][idx].item()),
                        "reconstruction_error": float(recon_error[idx].mean().item()),
                        "series_loss": float(series_loss[idx].mean().item()),
                        "prior_loss": float(prior_loss[idx].mean().item()),
                        "score": float(score[idx].item()),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the vanilla Anomaly Transformer and export raw scores")
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
    print(f"[validate_vanilla_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    checkpoint_path = run_dir / "models" / "vanilla_joint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    print(f"[validate_vanilla_joint] loading checkpoint={checkpoint_path}")

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
    print(f"[validate_vanilla_joint] windows total={len(manifest)} | val={len(val_ds)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")
    print(f"[validate_vanilla_joint] val_batches={len(val_loader)} | test_preview_batches={len(test_loader)}")

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

    print("[validate_vanilla_joint] scoring validation set", flush=True)
    val_df = collect_scores(model, val_loader, device, phase="validate", temperature=args.temperature)
    print("[validate_vanilla_joint] scoring test preview set", flush=True)
    test_df = collect_scores(model, test_loader, device, phase="test_preview", temperature=args.temperature)
    if val_df.empty:
        raise RuntimeError("Validation scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        reports_dir / "vanilla_joint_score_summary.json",
        {
            "prior_type": args.prior_type,
            "temperature": args.temperature,
            "val_score_mean": float(val_df["score"].mean()),
            "val_score_std": float(val_df["score"].std(ddof=0)),
            "val_score_min": float(val_df["score"].min()),
            "val_score_max": float(val_df["score"].max()),
            "test_preview_score_mean": float(test_df["score"].mean()),
            "test_preview_score_std": float(test_df["score"].std(ddof=0)),
            "test_preview_score_min": float(test_df["score"].min()),
            "test_preview_score_max": float(test_df["score"].max()),
            "val_windows": int(len(val_df)),
            "test_preview_windows": int(len(test_df)),
            "ticker_count": int(manifest["ticker"].nunique()),
            "split_counts": manifest["split"].value_counts().to_dict(),
        },
    )

    val_df.to_csv(reports_dir / "vanilla_joint_validation_scores.csv", index=False)
    test_df.to_csv(reports_dir / "vanilla_joint_test_scores_preview.csv", index=False)
    print(f"Saved raw score summary to {reports_dir / 'vanilla_joint_score_summary.json'}")
    print(f"Validation windows: {len(val_df)} | Test preview windows: {len(test_df)}")


if __name__ == "__main__":
    main()
