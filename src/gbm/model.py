from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gbm.embed import DataEmbedding


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


def symmetric_kl(series: torch.Tensor, prior: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    series = torch.clamp(series, min=eps)
    prior = torch.clamp(prior, min=eps)
    series_to_prior = torch.sum(series * (torch.log(series) - torch.log(prior)), dim=-1)
    prior_to_series = torch.sum(prior * (torch.log(prior) - torch.log(series)), dim=-1)
    return 0.5 * (series_to_prior + prior_to_series)


def temporal_prior(batch_size: int, n_heads: int, length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    distances = torch.abs(positions[None, :] - positions[:, None]).float()
    logits = -distances
    mask = causal_mask(length, device)
    logits = logits.masked_fill(mask, -1e9)
    prior = torch.softmax(logits, dim=-1)
    return prior.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, length, length)


def gaussian_log_return_attention_prior(
    returns: torch.Tensor,
    log_drift: torch.Tensor,
    sigma: torch.Tensor,
    n_heads: int,
    time_deltas: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Latent timestamp posterior from Gaussian log-price transitions.

    The drift parameter is a log-return drift alpha_t, so the transition mean
    over j -> i is sum alpha_u * dt_u across the interval.
    """
    return gaussian_transition_timestamp_posterior(
        returns,
        log_drift,
        sigma,
        n_heads,
        time_deltas=time_deltas,
        eps=eps,
    )


def canonical_gbm_attention_prior(
    returns: torch.Tensor,
    price_drift: torch.Tensor,
    sigma: torch.Tensor,
    n_heads: int,
    time_deltas: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Latent timestamp posterior from the canonical GBM price process.

    For dS_t = mu_t S_t dt + sigma_t S_t dW_t, the log-price transition drift
    is mu_t - 0.5 sigma_t^2. This function applies that Ito correction before
    accumulating the interval transition density.

    The returned matrix is p(J_i = j | Delta L_{j->i}, theta), where J_i is an
    explicit latent source timestamp with a uniform prior over j < i. The
    diagonal i == j is excluded from the continuous GBM density; only the first
    row falls back to self mass because there is no past timestamp.
    """
    sigma = torch.clamp(sigma, min=eps)
    log_drift = price_drift - 0.5 * sigma**2
    return gaussian_transition_timestamp_posterior(
        returns,
        log_drift,
        sigma,
        n_heads,
        time_deltas=time_deltas,
        eps=eps,
    )


def gaussian_transition_timestamp_posterior(
    returns: torch.Tensor,
    interval_log_drift: torch.Tensor,
    sigma: torch.Tensor,
    n_heads: int,
    time_deltas: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, length = returns.shape
    device = returns.device
    if time_deltas is None:
        time_deltas = torch.ones_like(returns)
    else:
        time_deltas = torch.clamp(time_deltas.to(device=device, dtype=returns.dtype), min=eps)

    log_level = torch.cumsum(returns, dim=1)
    increments = log_level[:, :, None] - log_level[:, None, :]

    positions = torch.arange(length, device=device)
    raw_dt = positions[:, None] - positions[None, :]
    past = raw_dt > 0

    sigma = torch.clamp(sigma, min=eps)
    dt = time_deltas[:, None, :]
    drift_prefix = F.pad(torch.cumsum(interval_log_drift * dt, dim=2), (1, 0))
    variance_prefix = F.pad(torch.cumsum(sigma**2 * dt, dim=2), (1, 0))
    mean = drift_prefix[:, :, 1:, None] - drift_prefix[:, :, None, 1:]
    variance = variance_prefix[:, :, 1:, None] - variance_prefix[:, :, None, 1:]
    variance = torch.clamp(variance, min=eps)
    std = torch.sqrt(variance)

    z = (increments[:, None, :, :] - mean) / std
    log_prob = -0.5 * (z**2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    candidate_prior = torch.full((length, length), -1e9, device=device, dtype=returns.dtype)
    past_counts = torch.arange(length, device=device, dtype=returns.dtype).clamp(min=1.0)
    candidate_prior = candidate_prior.masked_fill(past, 0.0)
    candidate_prior = candidate_prior - torch.log(past_counts[:, None])
    posterior_logits = log_prob + candidate_prior[None, None, :, :]
    posterior = torch.softmax(posterior_logits, dim=-1)
    first_row = torch.zeros(length, device=device, dtype=returns.dtype)
    first_row[0] = 1.0
    row0_mask = (positions == 0)[None, None, :, None]
    return torch.where(row0_mask, first_row[None, None, None, :], posterior)


gaussian_transition_attention_prior = gaussian_transition_timestamp_posterior


class TransitionPriorEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        association_mode: str = "gaussian_log_return",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if association_mode not in {"gaussian_log_return", "canonical_gbm", "temporal", "none"}:
            raise ValueError("association_mode must be gaussian_log_return, canonical_gbm, temporal, or none")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.association_mode = association_mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.mu_prior = nn.Linear(d_model, n_heads)
        self.sigma_prior = nn.Linear(d_model, n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, length, d_model = x.shape
        q = self.q_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None, :, :], -1e9)
        series = torch.softmax(scores, dim=-1)
        attn_values = torch.matmul(self.attn_dropout(series), v)
        attn_out = self.out_proj(attn_values.transpose(1, 2).contiguous().view(batch_size, length, d_model))

        prior_drift = self.mu_prior(x).transpose(1, 2)
        prior_sigma = F.softplus(self.sigma_prior(x)).transpose(1, 2) + 1e-4
        if self.association_mode == "none":
            prior = series.detach()
            discrepancy = torch.zeros(batch_size, device=x.device)
        elif self.association_mode == "temporal" or returns is None:
            prior = temporal_prior(batch_size, self.n_heads, length, x.device)
            discrepancy = symmetric_kl(series, prior).mean(dim=(1, 2))
        else:
            prior_fn = (
                canonical_gbm_attention_prior
                if self.association_mode == "canonical_gbm"
                else gaussian_log_return_attention_prior
            )
            prior = prior_fn(
                returns,
                prior_drift,
                prior_sigma,
                self.n_heads,
                time_deltas=time_deltas,
            )
            discrepancy = symmetric_kl(series, prior).mean(dim=(1, 2))

        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, {
            "series": series,
            "prior": prior,
            "discrepancy": discrepancy,
            "prior_drift": prior_drift,
            "prior_sigma": prior_sigma,
        }


class AnomalyTransformer(nn.Module):
    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        predictive_distribution: str = "gaussian",
        association_mode: str = "gaussian_log_return",
    ):
        super().__init__()
        if predictive_distribution not in {"gaussian", "student_t"}:
            raise ValueError("predictive_distribution must be gaussian or student_t")
        if association_mode not in {"gaussian_log_return", "canonical_gbm", "temporal", "none"}:
            raise ValueError("association_mode must be gaussian_log_return, canonical_gbm, temporal, or none")
        self.win_size = win_size
        self.enc_in = enc_in
        self.c_out = c_out
        self.predictive_distribution = predictive_distribution
        self.association_mode = association_mode
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                TransitionPriorEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    association_mode=association_mode,
                )
                for _ in range(e_layers)
            ]
        )
        summary_dim = d_model * 2
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.nu_head = nn.Sequential(
            nn.Linear(summary_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        hidden = self.embedding(x)
        attn_mask = causal_mask(hidden.shape[1], hidden.device)
        attn_maps: List[Dict[str, torch.Tensor]] = []

        for layer in self.layers:
            hidden, attn = layer(hidden, returns=returns, time_deltas=time_deltas, attn_mask=attn_mask)
            attn_maps.append(attn)

        recon = self.recon_head(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
        mu = self.mu_head(pooled).squeeze(-1)
        sigma = F.softplus(self.sigma_head(pooled)).squeeze(-1) + 1e-4
        nu = None
        if self.predictive_distribution == "student_t":
            nu = F.softplus(self.nu_head(pooled)).squeeze(-1) + 2.1

        obs_mu = None
        obs_sigma = None
        if returns is not None:
            obs_mu = returns.mean(dim=1)
            obs_sigma = returns.std(dim=1, unbiased=False) + 1e-4

        association_discrepancy = None
        if attn_maps:
            association_discrepancy = torch.stack([attn["discrepancy"] for attn in attn_maps], dim=0).mean(dim=0)

        if return_attention:
            return recon, mu, sigma, nu, attn_maps, obs_mu, obs_sigma, hidden, association_discrepancy
        return recon, mu, sigma, nu, obs_mu, obs_sigma, association_discrepancy


class GaussianLogReturnAttentionTransformer(AnomalyTransformer):
    def __init__(self, *args, association_mode: str = "gaussian_log_return", **kwargs):
        if association_mode != "gaussian_log_return":
            raise ValueError("GaussianLogReturnAttentionTransformer requires association_mode='gaussian_log_return'")
        super().__init__(*args, association_mode="gaussian_log_return", **kwargs)


class CanonicalGBMAttentionTransformer(AnomalyTransformer):
    def __init__(self, *args, association_mode: str = "canonical_gbm", **kwargs):
        if association_mode != "canonical_gbm":
            raise ValueError("CanonicalGBMAttentionTransformer requires association_mode='canonical_gbm'")
        super().__init__(*args, association_mode="canonical_gbm", **kwargs)
