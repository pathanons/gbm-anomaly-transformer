"""Canonical training surface for GBM experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.gbm.datasets import build_joint_loaders, discover_tickers, get_run_dir, save_joint_manifest, save_json, set_seed
from src.gbm.model import AnomalyTransformer
from src.gbm.score import (
    association_discrepancy_by_time,
    gaussian_wasserstein,
    mle_parameter_errors,
    normalized_moment_discrepancy,
    predictive_nll,
    predictive_std,
    softmax_association_weighted_nll,
)

def is_mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Resolve cuda/mps/cpu with an Apple Silicon friendly auto mode."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable; falling back to CPU", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")

    if requested == "mps":
        if not is_mps_available():
            print("[device] MPS requested but unavailable; falling back to CPU", flush=True)
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(requested)


def load_state_dict(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_tickers(data_path: str, tickers: list[str] | None) -> list[str]:
    return tickers if tickers else discover_tickers(data_path)


def build_runtime_loaders(args: Any):
    return build_joint_loaders(
        data_path=args.data_path,
        tickers=resolve_tickers(args.data_path, args.tickers),
        window_size=args.window_size,
        batch_size=args.batch_size,
        step=args.step,
        features=args.features,
        normalize_batch=args.normalize_batch,
        seed=args.seed,
        split_method=args.split_method,
        purge_gap=args.purge_gap,
        train_normal_only=not args.include_anomalous_train,
        target=getattr(args, "target", "window_returns"),
    )


def build_gbm_model(args: Any, input_dim: int, device) -> AnomalyTransformer:
    return AnomalyTransformer(
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


def checkpoint_path(args: Any, filename: str = "gbm.pt") -> Path:
    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    checkpoint_exp_name = getattr(args, "checkpoint_exp_name", None)
    checkpoint_run_dir = get_run_dir(checkpoint_exp_name, getattr(args, "output_root", None)) if checkpoint_exp_name else run_dir
    return checkpoint_run_dir / "models" / filename


def score_kwargs(args: Any) -> dict[str, Any]:
    return {
        "dist_weight": args.dist_weight,
        "recon_weight": args.recon_weight,
        "divergence_weight": args.divergence_weight,
        "association_weight": args.association_weight,
        "predictive_distribution": args.predictive_distribution,
        "score_mode": args.score_mode,
        "quantile_count": args.quantile_count,
        "tail_weight_gamma": args.tail_weight_gamma,
        "tail_weight_power": args.tail_weight_power,
    }


def nll_returns_for_batch(batch: dict[str, Any], returns: torch.Tensor, device) -> torch.Tensor:
    if "target_return" not in batch:
        return returns
    return batch["target_return"].to(device).unsqueeze(-1)


def run_gbm_epoch(
    model,
    loader,
    device,
    optimizer=None,
    dist_weight=1.0,
    recon_weight=1.0,
    divergence_weight=0.25,
    association_weight=0.1,
    predictive_distribution="gaussian",
    loss_mode="legacy",
    loss_combine_mode="weighted_sum",
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()
    batch_total = len(loader)
    phase = "train" if is_train else "val"

    context = torch.enable_grad() if is_train else torch.no_grad()
    use_recon = recon_weight != 0
    use_divergence = divergence_weight != 0
    use_association = association_weight != 0
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            if loss_combine_mode == "mle_param_minimax":
                if not use_association:
                    raise ValueError("loss_combine_mode=mle_param_minimax requires association_weight != 0")

                def mle_minimax_forward(detach_series: bool = False, detach_prior: bool = False):
                    recon_f, mu_f, sigma_f, nu_f, attn_maps_f, obs_mu_f, obs_sigma_f, latent_f, association_f = model(
                        x,
                        returns=returns,
                        time_deltas=time_deltas,
                        return_attention=True,
                    )
                    param_error_f, param_error_time_f, mu_real_f, sigma_real_f = mle_parameter_errors(
                        mu_f,
                        sigma_f,
                        attn_maps_f,
                        returns,
                        time_deltas=time_deltas,
                    )
                    association_time_f = association_discrepancy_by_time(
                        attn_maps_f,
                        detach_series=detach_series,
                        detach_prior=detach_prior,
                    )
                    return param_error_f.mean(), association_time_f.abs().mean()

                if is_train:
                    param_loss, association_loss = mle_minimax_forward(detach_series=True)
                    minimize_loss = dist_weight * param_loss - association_weight * association_loss
                    optimizer.zero_grad()
                    minimize_loss.backward()
                    optimizer.step()

                    param_loss, association_loss = mle_minimax_forward(detach_prior=True)
                    loss = dist_weight * param_loss + association_weight * association_loss
                else:
                    param_loss, association_loss = mle_minimax_forward(detach_prior=True)
                    loss = dist_weight * param_loss + association_weight * association_loss

                if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                    print(
                        f"    [{phase}] batch {batch_idx}/{batch_total} | loss={float(loss.item()):.6f} "
                        f"| mle_param={float(param_loss.item()):.6f} | assoc={float(association_loss.item()):.6f} "
                        f"| lambda={association_weight:g} | mode={loss_mode} | combine={loss_combine_mode}",
                        flush=True,
                    )
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item())
                count += 1
                continue

            if loss_combine_mode == "nll_assoc_minimax":
                if not use_association:
                    raise ValueError("loss_combine_mode=nll_assoc_minimax requires association_weight != 0")

                def minimax_forward(detach_series: bool = False, detach_prior: bool = False):
                    recon_f, mu_f, sigma_f, nu_f, attn_maps_f, obs_mu_f, obs_sigma_f, latent_f, association_f = model(
                        x,
                        returns=returns,
                        time_deltas=time_deltas,
                        return_attention=True,
                    )
                    nll_time_f = predictive_nll(nll_returns_for_batch(batch, returns, device), mu_f, sigma_f, nu_f, distribution=predictive_distribution)
                    nll_f = nll_time_f.mean()
                    association_time_f = association_discrepancy_by_time(
                        attn_maps_f,
                        detach_series=detach_series,
                        detach_prior=detach_prior,
                    )
                    association_abs_f = association_time_f.abs().mean()
                    recon_error_f = criterion(recon_f, x) if use_recon else torch.zeros((), device=device)
                    divergence_f = torch.zeros((), device=device)
                    if use_divergence:
                        pred_std_f = predictive_std(sigma_f, nu_f, distribution=predictive_distribution)
                        legacy_divergence_f = gaussian_wasserstein(obs_mu_f, obs_sigma_f, mu_f, pred_std_f).mean()
                        refactored_divergence_f = normalized_moment_discrepancy(
                            obs_mu_f,
                            obs_sigma_f,
                            mu_f,
                            pred_std_f,
                            window_length=int(returns.size(-1)),
                        ).mean()
                        divergence_f = legacy_divergence_f if loss_mode == "legacy" else refactored_divergence_f
                    return recon_error_f, nll_f, divergence_f, association_abs_f

                if is_train:
                    recon_error, nll, divergence, association_loss = minimax_forward(detach_series=False, detach_prior=False)
                    minimize_loss = (
                        (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                        + dist_weight * nll
                        + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                        + association_weight * association_loss
                    )
                    optimizer.zero_grad()
                    minimize_loss.backward()
                    optimizer.step()

                    recon_error, nll, divergence, association_loss = minimax_forward(detach_prior=True)
                    loss = (
                        (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                        + dist_weight * nll
                        + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                        - association_weight * association_loss
                    )
                else:
                    recon_error, nll, divergence, association_loss = minimax_forward(detach_prior=True)
                    loss = (
                        (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                        + dist_weight * nll
                        + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                        - association_weight * association_loss
                    )
                if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                    recon_text = f"recon={float(recon_error.item()):.6f}" if use_recon else "recon=off"
                    div_text = f"div={float(divergence.item()):.6f}" if use_divergence else "div=off"
                    assoc_text = f"assoc={float(association_loss.item()):.6f}"
                    print(
                        f"    [{phase}] batch {batch_idx}/{batch_total} | loss={float(loss.item()):.6f} "
                        f"| {recon_text} | nll={float(nll.item()):.6f} "
                        f"| {div_text} | {assoc_text} "
                        f"| mode={loss_mode} | combine={loss_combine_mode}",
                        flush=True,
                    )
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item())
                count += 1
                continue

            use_softmax_product = loss_combine_mode == "nll_assoc_softmax_product"
            if use_softmax_product:
                recon, mu, sigma, nu, attn_maps, obs_mu, obs_sigma, latent, association = model(
                    x,
                    returns=returns,
                    time_deltas=time_deltas,
                    return_attention=True,
                )
            else:
                recon, mu, sigma, nu, obs_mu, obs_sigma, association = model(
                    x,
                    returns=returns,
                    time_deltas=time_deltas,
                )
            recon_error = criterion(recon, x) if use_recon else torch.zeros((), device=device)
            nll_by_time = predictive_nll(nll_returns_for_batch(batch, returns, device), mu, sigma, nu, distribution=predictive_distribution)
            nll = nll_by_time.mean()
            if use_divergence:
                pred_std = predictive_std(sigma, nu, distribution=predictive_distribution)
                legacy_divergence = gaussian_wasserstein(obs_mu, obs_sigma, mu, pred_std).mean()
                refactored_divergence = normalized_moment_discrepancy(
                    obs_mu,
                    obs_sigma,
                    mu,
                    pred_std,
                    window_length=int(returns.size(-1)),
                ).mean()
                divergence = legacy_divergence if loss_mode == "legacy" else refactored_divergence
            else:
                divergence = torch.zeros((), device=device)
            association_loss = association.mean() if (use_association and association is not None) else torch.zeros((), device=device)
            if loss_combine_mode == "weighted_sum":
                loss = (
                    (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                    + dist_weight * nll
                    + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                    + (association_weight * association_loss if use_association else torch.zeros((), device=device))
                )
            elif loss_combine_mode == "nll_assoc_product":
                if not use_association:
                    raise ValueError("loss_combine_mode=nll_assoc_product requires association_weight != 0")
                loss = (
                    (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                    + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                    + dist_weight * association_weight * nll * association_loss
                )
            elif loss_combine_mode == "nll_assoc_softmax_product":
                if not use_association:
                    raise ValueError("loss_combine_mode=nll_assoc_softmax_product requires association_weight != 0")
                weighted_nll, _, _ = softmax_association_weighted_nll(
                    nll_by_time,
                    attn_maps,
                    association_weight=association_weight,
                )
                loss = (
                    (recon_weight * recon_error if use_recon else torch.zeros((), device=device))
                    + (divergence_weight * divergence if use_divergence else torch.zeros((), device=device))
                    + dist_weight * weighted_nll.mean()
                )
            else:
                raise ValueError("loss_combine_mode must be weighted_sum, nll_assoc_product, nll_assoc_softmax_product, nll_assoc_minimax, or mle_param_minimax")

            if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                recon_text = f"recon={float(recon_error.item()):.6f}" if use_recon else "recon=off"
                div_text = f"div={float(divergence.item()):.6f}" if use_divergence else "div=off"
                assoc_text = f"assoc={float(association_loss.item()):.6f}" if use_association else "assoc=off"
                print(
                    f"    [{phase}] batch {batch_idx}/{batch_total} | loss={float(loss.item()):.6f} "
                    f"| {recon_text} | nll={float(nll.item()):.6f} "
                    f"| {div_text} | {assoc_text} "
                    f"| mode={loss_mode} | combine={loss_combine_mode}",
                    flush=True,
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            count += 1
    return total_loss / max(1, count)


def train_model(args: Any) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[train] device={device}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    manifest_path = save_joint_manifest(
        run_dir,
        manifest,
        args.seed,
        args.window_size,
        args.step,
        args.features,
        split_method=args.split_method,
        purge_gap=args.purge_gap,
        train_normal_only=not args.include_anomalous_train,
        target=getattr(args, "target", "window_returns"),
    )

    split_counts = manifest["split"].value_counts().to_dict()
    print(f"[train] windows={len(manifest)} | split_counts={split_counts}")
    print(f"[train] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[train] manifest={manifest_path}")
    print(f"[train] train_batches={len(train_loader)} | val_batches={len(val_loader)} | test_batches={len(test_loader)}")
    print(f"[train] tickers={manifest['ticker'].nunique()}", flush=True)

    model = build_gbm_model(args, input_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    patience_count = 0
    checkpoint_dir = run_dir / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = "gbm.pt"
    checkpoint_path = checkpoint_dir / model_name

    history = []
    for epoch in range(args.epochs):
        print(f"[train] epoch {epoch + 1}/{args.epochs} starting", flush=True)
        train_loss = run_gbm_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            dist_weight=args.dist_weight,
            recon_weight=args.recon_weight,
            divergence_weight=args.divergence_weight,
            association_weight=args.association_weight,
            predictive_distribution=args.predictive_distribution,
            loss_mode=args.loss_mode,
            loss_combine_mode=getattr(args, "loss_combine_mode", "weighted_sum"),
        )
        val_loss = run_gbm_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            dist_weight=args.dist_weight,
            recon_weight=args.recon_weight,
            divergence_weight=args.divergence_weight,
            association_weight=args.association_weight,
            predictive_distribution=args.predictive_distribution,
            loss_mode=args.loss_mode,
            loss_combine_mode=getattr(args, "loss_combine_mode", "weighted_sum"),
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    save_json(run_dir / "logs" / "training.json", history)
    config = {
        **vars(args),
        "input_dim": input_dim,
        "best_val_loss": best_val,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
    }
    save_json(run_dir / "configs" / "config.json", config)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved split manifest to {manifest_path}")
    return config

__all__ = [
    "build_gbm_model",
    "build_runtime_loaders",
    "run_gbm_epoch",
    "train_model",
]

__all__ = [
    "build_gbm_model",
    "build_runtime_loaders",
    "checkpoint_path",
    "is_mps_available",
    "load_state_dict",
    "resolve_device",
    "resolve_tickers",
    "run_gbm_epoch",
    "score_kwargs",
    "train_model",
]
