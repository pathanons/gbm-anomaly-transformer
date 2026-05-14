from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gbm.embed import DataEmbedding


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class GBMEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


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
        attn_maps: List[torch.Tensor] = []

        for layer in self.layers:
            hidden, attn = layer(hidden, attn_mask=attn_mask)
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

        if return_attention:
            return recon, mu, sigma, attn_maps, obs_mu, obs_sigma, hidden
        return recon, mu, sigma, obs_mu, obs_sigma


GBMAnomalyTransformer = AnomalyTransformer
