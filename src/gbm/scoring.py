from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from src.gbm.losses import gaussian_nll, gaussian_wasserstein, score_windows


def collect_joint_scores(
    model,
    loader,
    device,
    phase: str,
    recon_weight: float = 0.25,
    association_weight: float = 0.1,
) -> pd.DataFrame:
    model.eval()
    rows = []
    criterion = nn.MSELoss(reduction="none")
    batch_total = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            recon, mu, sigma, attn_maps, obs_mu, obs_sigma, latent, association = model(
                x,
                returns=returns,
                return_attention=True,
            )
            recon_error = criterion(recon, x).mean(dim=(1, 2))
            nll = gaussian_nll(returns, mu, sigma).mean(dim=1)
            divergence = gaussian_wasserstein(obs_mu, obs_sigma, mu, sigma)
            score = score_windows(
                recon_error,
                nll,
                divergence,
                association=association,
                recon_weight=recon_weight,
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
                        "mu_pred": float(mu[idx].item()),
                        "sigma_pred": float(sigma[idx].item()),
                        "mu_obs": float(obs_mu[idx].item()),
                        "sigma_obs": float(obs_sigma[idx].item()),
                    }
                )

    return pd.DataFrame(rows)
