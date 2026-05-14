#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

from src.gbm.data import build_joint_loaders, discover_tickers, get_run_dir, save_joint_manifest, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.losses import gaussian_nll, gaussian_wasserstein, score_windows
from src.gbm.model import AnomalyTransformer


def run_epoch(model, loader, device, optimizer=None, dist_weight=1.0, recon_weight=1.0, divergence_weight=0.25):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()
    batch_total = len(loader)
    phase = "train" if is_train else "val"

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            recon, mu, sigma, obs_mu, obs_sigma = model(x, returns=returns)
            recon_error = criterion(recon, x)
            nll = gaussian_nll(returns, mu, sigma).mean()
            divergence = gaussian_wasserstein(obs_mu, obs_sigma, mu, sigma).mean()
            loss = recon_weight * recon_error + dist_weight * nll + divergence_weight * divergence

            if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                print(
                    f"    [{phase}] batch {batch_idx}/{batch_total} | loss={float(loss.item()):.6f} "
                    f"| recon={float(recon_error.item()):.6f} | nll={float(nll.item()):.6f} | div={float(divergence.item()):.6f}",
                    flush=True,
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            count += 1
    return total_loss / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single EXP3 GBM model on all tickers jointly")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", default="experiment3_joint")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[train_joint] device={device}", flush=True)
    run_dir = get_run_dir(args.exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)

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
    manifest_path = save_joint_manifest(run_dir, manifest, args.seed, args.window_size, args.step, args.features)

    split_counts = manifest["split"].value_counts().to_dict()
    print(f"[train_joint] windows={len(manifest)} | split_counts={split_counts}")
    print(f"[train_joint] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[train_joint] manifest={manifest_path}")
    print(f"[train_joint] train_loader_batches={len(train_loader)} | val_loader_batches={len(val_loader)} | test_loader_batches={len(test_loader)}")
    print(f"[train_joint] tickers={manifest['ticker'].nunique()}", flush=True)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    patience = 0
    checkpoint_dir = run_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "gbm_joint.pt"

    history = []
    for epoch in range(args.epochs):
        print(f"[train_joint] epoch {epoch + 1}/{args.epochs} starting", flush=True)
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            dist_weight=args.dist_weight,
            recon_weight=args.recon_weight,
            divergence_weight=args.divergence_weight,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            dist_weight=args.dist_weight,
            recon_weight=args.recon_weight,
            divergence_weight=args.divergence_weight,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    save_json(run_dir / "logs" / "gbm_joint_training.json", history)
    save_json(
        run_dir / "configs" / "gbm_joint_config.json",
        {
            **vars(args),
            "input_dim": input_dim,
            "best_val_loss": best_val,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "test_windows": len(test_ds),
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": str(manifest_path),
        },
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved split manifest to {manifest_path}")


if __name__ == "__main__":
    main()
