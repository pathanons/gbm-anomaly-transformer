from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn

from src.gbm.losses import (
    gaussian_wasserstein,
    normalized_moment_discrepancy,
    predictive_nll,
    predictive_std,
    quantile_wasserstein_score,
    score_windows,
    tail_weighted_quantile_wasserstein_score,
)


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
    score_mode: str = "legacy",
    quantile_count: int = 21,
    tail_weight_gamma: float = 2.0,
    tail_weight_power: float = 2.0,
) -> pd.DataFrame:
    model.eval()
    rows = []
    criterion = nn.MSELoss(reduction="none")
    batch_total = len(loader)
    use_recon = recon_weight != 0
    use_divergence = divergence_weight != 0
    use_association = association_weight != 0

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
            recon_error = criterion(recon, x).mean(dim=(1, 2)) if use_recon else torch.zeros(x.size(0), device=device)
            nll = predictive_nll(returns, mu, sigma, nu, distribution=predictive_distribution).mean(dim=1)
            pred_std = predictive_std(sigma, nu, distribution=predictive_distribution)
            window_length = int(returns.size(-1))
            if use_divergence:
                legacy_divergence = gaussian_wasserstein(obs_mu, obs_sigma, mu, pred_std)
                refactored_divergence = normalized_moment_discrepancy(obs_mu, obs_sigma, mu, pred_std, window_length)
                dist_qw2 = quantile_wasserstein_score(
                    returns,
                    mu,
                    pred_std,
                    nu=nu,
                    distribution=predictive_distribution,
                    quantile_count=quantile_count,
                )
                dist_qw2_tail = tail_weighted_quantile_wasserstein_score(
                    returns,
                    mu,
                    pred_std,
                    nu=nu,
                    distribution=predictive_distribution,
                    quantile_count=quantile_count,
                    gamma=tail_weight_gamma,
                    power=tail_weight_power,
                )
            else:
                legacy_divergence = torch.zeros_like(nll)
                refactored_divergence = torch.zeros_like(nll)
                dist_qw2 = torch.zeros_like(nll)
                dist_qw2_tail = torch.zeros_like(nll)
            legacy_score = score_windows(
                recon_error,
                nll,
                legacy_divergence,
                association=association if use_association else None,
                dist_weight=dist_weight,
                recon_weight=recon_weight,
                divergence_weight=divergence_weight,
                association_weight=association_weight,
            )
            refactored_score = score_windows(
                recon_error,
                nll,
                refactored_divergence,
                association=association if use_association else None,
                dist_weight=dist_weight,
                recon_weight=recon_weight,
                divergence_weight=divergence_weight,
                association_weight=association_weight,
            )
            score_qw2 = score_windows(
                recon_error,
                nll,
                dist_qw2,
                association=association if use_association else None,
                dist_weight=dist_weight,
                recon_weight=recon_weight,
                divergence_weight=divergence_weight,
                association_weight=association_weight,
            )
            score_qw2_tail = score_windows(
                recon_error,
                nll,
                dist_qw2_tail,
                association=association if use_association else None,
                dist_weight=dist_weight,
                recon_weight=recon_weight,
                divergence_weight=divergence_weight,
                association_weight=association_weight,
            )
            if score_mode == "legacy":
                score = legacy_score
            elif score_mode == "refactored":
                score = refactored_score
            elif score_mode == "qw2":
                score = score_qw2
            elif score_mode == "qw2_tail":
                score = score_qw2_tail
            else:
                raise ValueError("score_mode must be legacy, refactored, qw2, or qw2_tail")

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
                        "divergence": float(legacy_divergence[idx].item()),
                        "refactored_divergence": float(refactored_divergence[idx].item()),
                        "dist_qw2": float(dist_qw2[idx].item()),
                        "dist_qw2_tail": float(dist_qw2_tail[idx].item()),
                        "association_discrepancy": float(association[idx].item()) if (use_association and association is not None) else 0.0,
                        "score": float(score[idx].item()),
                        "legacy_score": float(legacy_score[idx].item()),
                        "refactored_score": float(refactored_score[idx].item()),
                        "score_qw2": float(score_qw2[idx].item()),
                        "score_qw2_tail": float(score_qw2_tail[idx].item()),
                        "score_mode": score_mode,
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
