"""Canonical loss and scoring surface for GBM experiments."""
from __future__ import annotations

import math
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

def gaussian_nll(returns: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=1e-4)
    mu = mu.unsqueeze(-1)
    sigma = sigma.unsqueeze(-1)
    return 0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma) + ((returns - mu) ** 2) / (sigma ** 2))


def student_t_nll(returns: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    scale = torch.clamp(scale, min=1e-4)
    nu = torch.clamp(nu, min=2.1)
    mu = mu.unsqueeze(-1)
    scale = scale.unsqueeze(-1)
    nu = nu.unsqueeze(-1)
    z = (returns - mu) / scale
    log_norm = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * (torch.log(nu) + math.log(math.pi))
        - torch.log(scale)
    )
    log_kernel = -0.5 * (nu + 1.0) * torch.log1p((z**2) / nu)
    return -(log_norm + log_kernel)


def predictive_nll(
    returns: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | None = None,
    distribution: str = "gaussian",
) -> torch.Tensor:
    if distribution == "gaussian":
        return gaussian_nll(returns, mu, sigma)
    if distribution == "student_t":
        if nu is None:
            raise ValueError("Student-t predictive NLL requires nu")
        return student_t_nll(returns, mu, sigma, nu)
    raise ValueError("distribution must be gaussian or student_t")


def predictive_std(sigma: torch.Tensor, nu: torch.Tensor | None = None, distribution: str = "gaussian") -> torch.Tensor:
    sigma = torch.clamp(sigma, min=1e-4)
    if distribution == "gaussian":
        return sigma
    if distribution == "student_t":
        if nu is None:
            raise ValueError("Student-t predictive std requires nu")
        nu = torch.clamp(nu, min=2.1)
        return sigma * torch.sqrt(nu / (nu - 2.0))
    raise ValueError("distribution must be gaussian or student_t")


def gaussian_wasserstein(mu_a: torch.Tensor, sigma_a: torch.Tensor, mu_b: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    sigma_a = torch.clamp(sigma_a, min=1e-4)
    sigma_b = torch.clamp(sigma_b, min=1e-4)
    return (mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2


def normalized_moment_discrepancy(
    mu_obs: torch.Tensor,
    sigma_obs: torch.Tensor,
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    window_length: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    sigma_obs = torch.clamp(sigma_obs, min=1e-4)
    sigma_pred = torch.clamp(sigma_pred, min=1e-4)
    window_length = max(int(window_length), 1)
    mean_term = (mu_obs - mu_pred) ** 2 / (sigma_pred**2 / float(window_length) + eps)
    vol_term = torch.log((sigma_obs + eps) / (sigma_pred + eps)) ** 2
    return mean_term + vol_term


def _gaussian_quantiles(mu: torch.Tensor, sigma: torch.Tensor, quantile_levels: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=1e-4)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * quantile_levels - 1.0)
    return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z.unsqueeze(0)


def _student_t_quantiles_mc(
    mu: torch.Tensor,
    scale: torch.Tensor,
    nu: torch.Tensor,
    quantile_levels: torch.Tensor,
    sample_size: int,
) -> torch.Tensor:
    scale = torch.clamp(scale, min=1e-4)
    nu = torch.clamp(nu, min=2.1)
    distribution = torch.distributions.StudentT(df=nu, loc=mu, scale=scale)
    samples = distribution.sample((max(int(sample_size), 64),))  # [M, B]
    q = torch.quantile(samples, quantile_levels, dim=0)
    return q.transpose(0, 1)


def quantile_wasserstein_score(
    returns: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | None = None,
    distribution: str = "gaussian",
    quantile_count: int = 21,
) -> torch.Tensor:
    quantile_count = max(int(quantile_count), 3)
    levels = torch.linspace(0.01, 0.99, steps=quantile_count, device=returns.device, dtype=returns.dtype)
    observed_quantiles = torch.quantile(returns, levels, dim=1).transpose(0, 1)

    if distribution == "gaussian":
        predicted_quantiles = _gaussian_quantiles(mu, sigma, levels)
    elif distribution == "student_t":
        if nu is None:
            raise ValueError("Student-t quantile Wasserstein requires nu")
        predicted_quantiles = _student_t_quantiles_mc(mu, sigma, nu, levels, sample_size=returns.shape[1])
    else:
        raise ValueError("distribution must be gaussian or student_t")

    squared_error = (observed_quantiles - predicted_quantiles) ** 2
    return squared_error.mean(dim=1)


def tail_weighted_quantile_wasserstein_score(
    returns: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor | None = None,
    distribution: str = "gaussian",
    quantile_count: int = 21,
    gamma: float = 2.0,
    power: float = 2.0,
) -> torch.Tensor:
    quantile_count = max(int(quantile_count), 3)
    levels = torch.linspace(0.01, 0.99, steps=quantile_count, device=returns.device, dtype=returns.dtype)
    observed_quantiles = torch.quantile(returns, levels, dim=1).transpose(0, 1)

    if distribution == "gaussian":
        predicted_quantiles = _gaussian_quantiles(mu, sigma, levels)
    elif distribution == "student_t":
        if nu is None:
            raise ValueError("Student-t tail-weighted quantile Wasserstein requires nu")
        predicted_quantiles = _student_t_quantiles_mc(mu, sigma, nu, levels, sample_size=returns.shape[1])
    else:
        raise ValueError("distribution must be gaussian or student_t")

    squared_error = (observed_quantiles - predicted_quantiles) ** 2
    weights = 1.0 + float(gamma) * torch.abs(levels - 0.5) ** float(power)
    weighted_error = squared_error * weights.unsqueeze(0)
    return weighted_error.sum(dim=1) / torch.clamp(weights.sum(), min=1e-8)


def score_windows(
    recon_error: torch.Tensor,
    nll: torch.Tensor,
    divergence: torch.Tensor,
    association: torch.Tensor | None = None,
    dist_weight: float = 1.0,
    recon_weight: float = 1.0,
    divergence_weight: float = 0.25,
    association_weight: float = 0.1,
) -> torch.Tensor:
    score = dist_weight * nll + divergence_weight * divergence + recon_weight * recon_error
    if association is not None:
        score = score + association_weight * association
    return score


EVENT_COLORS = {
    "jump": "#2ca02c",
    "drop": "#9467bd",
    "volume_spike": "#17becf",
    "volatility_shock": "#8c564b",
    "regime_shift": "#e377c2",
    "is_anomaly": "#2ca02c",
}


def roc_auc_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    negatives = int((data["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = data["score"].rank(method="average")
    positive_rank_sum = float(ranks[data["y"] == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    if positives == 0:
        return float("nan")
    data = data.sort_values("score", ascending=False).reset_index(drop=True)
    true_positive_count = 0
    precision_sum = 0.0
    for idx, label in enumerate(data["y"], start=1):
        if int(label) == 1:
            true_positive_count += 1
            precision_sum += true_positive_count / idx
    return precision_sum / positives

def event_columns_from_label_file(label_df: pd.DataFrame) -> list[str]:
    event_columns = [column for column in label_df.columns if column not in {"Date", "is_anomaly"}]
    if not event_columns and "is_anomaly" in label_df.columns:
        event_columns = ["is_anomaly"]
    return event_columns


def add_test_event_columns(results: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    frames = []
    all_event_columns = set()
    for ticker, frame in results.groupby("ticker", sort=False):
        frame = frame.copy()
        label_path = data_path / f"{ticker}_anomaly_label.csv"
        if not label_path.exists():
            frames.append(frame)
            continue

        label_df = pd.read_csv(label_path, parse_dates=["Date"])
        event_columns = event_columns_from_label_file(label_df)
        all_event_columns.update(event_columns)
        for event_column in event_columns:
            event_dates = label_df.loc[label_df[event_column].fillna(0).astype(float) > 0, "Date"]
            event_dates = sorted(pd.to_datetime(event_dates.drop_duplicates()))
            values = []
            for _, row in frame.iterrows():
                start_date = pd.to_datetime(row["start_date"])
                end_date = pd.to_datetime(row["end_date"])
                event_idx = bisect_left(event_dates, start_date)
                values.append(int(event_idx < len(event_dates) and event_dates[event_idx] <= end_date))
            frame[f"true_{event_column}"] = values
        frames.append(frame)

    annotated = pd.concat(frames, ignore_index=True)
    for event_column in sorted(all_event_columns):
        output_column = f"true_{event_column}"
        if output_column not in annotated.columns:
            annotated[output_column] = 0
        annotated[output_column] = annotated[output_column].fillna(0).astype(int)
    return annotated


# Weights applied to GBM score components:
# score = dist*nll + recon*reconstruction_error + div*divergence + assoc*association_discrepancy
#
# association_discrepancy is the KL-based association term from Anomaly Transformer lineage.

ScoreWeights = Dict[str, float]

SCORE_VARIANTS: Dict[str, ScoreWeights] = {
    "recon_only": {
        "dist_weight": 0.0,
        "recon_weight": 1.0,
        "divergence_weight": 0.0,
        "association_weight": 0.0,
    },
    "association_kl_only": {
        "dist_weight": 0.0,
        "recon_weight": 0.0,
        "divergence_weight": 0.0,
        "association_weight": 1.0,
    },
    "nll_only": {
        "dist_weight": 1.0,
        "recon_weight": 0.0,
        "divergence_weight": 0.0,
        "association_weight": 0.0,
    },
    "divergence_only": {
        "dist_weight": 0.0,
        "recon_weight": 0.0,
        "divergence_weight": 1.0,
        "association_weight": 0.0,
    },
    "recon_plus_kl": {
        "dist_weight": 0.0,
        "recon_weight": 1.0,
        "divergence_weight": 0.0,
        "association_weight": 1.0,
    },
    "nll_plus_recon": {
        "dist_weight": 1.0,
        "recon_weight": 1.0,
        "divergence_weight": 0.0,
        "association_weight": 0.0,
    },
    "full_default": {
        "dist_weight": 1.0,
        "recon_weight": 1.0,
        "divergence_weight": 0.25,
        "association_weight": 0.1,
    },
}

REQUIRED_COMPONENT_COLUMNS = (
    "reconstruction_error",
    "nll",
    "divergence",
    "association_discrepancy",
)


def compute_variant_score(frame: pd.DataFrame, weights: ScoreWeights) -> pd.Series:
    missing = [column for column in REQUIRED_COMPONENT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing score component columns: {', '.join(missing)}")
    return (
        weights["dist_weight"] * frame["nll"].astype(float)
        + weights["recon_weight"] * frame["reconstruction_error"].astype(float)
        + weights["divergence_weight"] * frame["divergence"].astype(float)
        + weights["association_weight"] * frame["association_discrepancy"].astype(float)
    )


def apply_score_variants(frame: pd.DataFrame, variants: Dict[str, ScoreWeights] | None = None) -> pd.DataFrame:
    variants = variants or SCORE_VARIANTS
    output = frame.copy()
    for variant_name, weights in variants.items():
        output[f"score_{variant_name}"] = compute_variant_score(output, weights)
    return output


def variant_label(variant_name: str) -> str:
    labels = {
        "recon_only": "Reconstruction only",
        "association_kl_only": "Association KL only",
        "nll_only": "Predictive NLL only",
        "divergence_only": "Wasserstein divergence only",
        "recon_plus_kl": "Reconstruction + KL",
        "nll_plus_recon": "NLL + reconstruction",
        "full_default": "Full mixed (default weights)",
    }
    return labels.get(variant_name, variant_name)


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = (y_score_arr > threshold).astype(int)

    if len(np.unique(y_true_arr)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1_score": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


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

__all__ = [
    "EVENT_COLORS",
    "SCORE_VARIANTS",
    "ScoreWeights",
    "add_test_event_columns",
    "apply_score_variants",
    "average_precision_score_local",
    "binary_metrics",
    "collect_joint_scores",
    "compute_variant_score",
    "event_columns_from_label_file",
    "gaussian_nll",
    "gaussian_wasserstein",
    "normalized_moment_discrepancy",
    "predictive_nll",
    "predictive_std",
    "quantile_wasserstein_score",
    "roc_auc_score_local",
    "score_windows",
    "student_t_nll",
    "tail_weighted_quantile_wasserstein_score",
    "variant_label",
]
