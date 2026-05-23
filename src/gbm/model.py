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


def gbm_prior_association(
    returns: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    n_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch_size, length = returns.shape
    device = returns.device
    log_level = torch.cumsum(returns, dim=1)
    increments = log_level[:, :, None] - log_level[:, None, :]

    positions = torch.arange(length, device=device)
    raw_dt = positions[:, None] - positions[None, :]
    valid = raw_dt >= 0
    dt = raw_dt.clamp(min=1).float()

    mu = mu[:, :, :, None]
    sigma = torch.clamp(sigma[:, :, :, None], min=eps)
    mean = mu * dt[None, None, :, :]
    std = sigma * torch.sqrt(dt)[None, None, :, :]
    z = (increments[:, None, :, :] - mean) / std
    log_prob = -0.5 * (z**2) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    log_prob = log_prob.masked_fill(~valid[None, None, :, :], -1e9)
    return torch.softmax(log_prob, dim=-1)


class GBMEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
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

        prior_mu = self.mu_prior(x).transpose(1, 2)
        prior_sigma = F.softplus(self.sigma_prior(x)).transpose(1, 2) + 1e-4
        if returns is None:
            prior = temporal_prior(batch_size, self.n_heads, length, x.device)
        else:
            prior = gbm_prior_association(returns, prior_mu, prior_sigma, self.n_heads)
        discrepancy = symmetric_kl(series, prior).mean(dim=(1, 2))

        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, {
            "series": series,
            "prior": prior,
            "discrepancy": discrepancy,
            "prior_mu": prior_mu,
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
    ):
        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.c_out = c_out
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layers = nn.ModuleList(
            [GBMEncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(e_layers)]
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

    def forward(self, x: torch.Tensor, returns: Optional[torch.Tensor] = None, return_attention: bool = False):
        hidden = self.embedding(x)
        attn_mask = causal_mask(hidden.shape[1], hidden.device)
        attn_maps: List[Dict[str, torch.Tensor]] = []

        for layer in self.layers:
            hidden, attn = layer(hidden, returns=returns, attn_mask=attn_mask)
            attn_maps.append(attn)

        recon = self.recon_head(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
        mu = self.mu_head(pooled).squeeze(-1)
        sigma = F.softplus(self.sigma_head(pooled)).squeeze(-1) + 1e-4

        obs_mu = None
        obs_sigma = None
        if returns is not None:
            obs_mu = returns.mean(dim=1)
            obs_sigma = returns.std(dim=1, unbiased=False) + 1e-4

        association_discrepancy = None
        if attn_maps:
            association_discrepancy = torch.stack([attn["discrepancy"] for attn in attn_maps], dim=0).mean(dim=0)

        if return_attention:
            return recon, mu, sigma, attn_maps, obs_mu, obs_sigma, hidden, association_discrepancy
        return recon, mu, sigma, obs_mu, obs_sigma, association_discrepancy


GBMAnomalyTransformer = AnomalyTransformer
