from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from src.gbm.losses import gaussian_wasserstein, predictive_nll, predictive_std, score_windows


def collect_joint_scores(
    model,
    loader,
    device,
    phase: str,
    dist_weight: float = 1.0,
    recon_weight: float = 1.0,
    divergence_weight: float = 0.25,
    association_weight: float = 0.1,
    predictive_distribution: str = "gaussian",
) -> pd.DataFrame:
    model.eval()
    rows = []
    criterion = nn.MSELoss(reduction="none")
    batch_total = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            recon, mu, sigma, nu, attn_maps, obs_mu, obs_sigma, latent, association = model(
                x,
                returns=returns,
                time_deltas=time_deltas,
                return_attention=True,
            )
            recon_error = criterion(recon, x).mean(dim=(1, 2))
            nll = predictive_nll(returns, mu, sigma, nu, distribution=predictive_distribution).mean(dim=1)
            pred_std = predictive_std(sigma, nu, distribution=predictive_distribution)
            divergence = gaussian_wasserstein(obs_mu, obs_sigma, mu, pred_std)
            score = score_windows(
                recon_error,
                nll,
                divergence,
                association=association,
                dist_weight=dist_weight,
                recon_weight=recon_weight,
                divergence_weight=divergence_weight,
                association_weight=association_weight,
            )

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
                        "reconstruction_error": float(recon_error[idx].item()),
                        "nll": float(nll[idx].item()),
                        "divergence": float(divergence[idx].item()),
                        "association_discrepancy": float(association[idx].item()) if association is not None else 0.0,
                        "score": float(score[idx].item()),
                        "predictive_distribution": predictive_distribution,
                        "mu_pred": float(mu[idx].item()),
                        "sigma_pred": float(sigma[idx].item()),
                        "nu_pred": float(nu[idx].item()) if nu is not None else float("nan"),
                        "mu_obs": float(obs_mu[idx].item()),
                        "sigma_obs": float(obs_sigma[idx].item()),
                        "tail_z_abs": float((torch.abs(returns[idx] - mu[idx]) / pred_std[idx]).max().item()),
                    }
                )

    return pd.DataFrame(rows)
