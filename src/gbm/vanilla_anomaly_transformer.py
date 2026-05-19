from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gbm.embed import DataEmbedding


class TriangularCausalMask:
    def __init__(self, batch_size: int, length: int, device: torch.device | str = "cpu"):
        mask_shape = [batch_size, 1, length, length]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(
        self,
        win_size: int,
        mask_flag: bool = True,
        scale: float | None = None,
        attention_dropout: float = 0.0,
        output_attention: bool = False,
        prior_type: str = "gaussian",
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.prior_type = prior_type
        self.register_buffer("distances", torch.zeros((win_size, win_size), dtype=torch.float32), persistent=False)
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i, j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask, alpha=None):
        batch_size, length, heads, embed_dim = queries.shape
        scale = self.scale or 1.0 / math.sqrt(embed_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, length, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        window_size = attn.shape[-1]

        if self.prior_type == "gaussian":
            sigma = sigma.transpose(1, 2)
            sigma = torch.sigmoid(sigma * 5) + 1e-5
            sigma = torch.pow(3, sigma) - 1
            sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)
            prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(sigma.device)
            prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior**2 / 2 / (sigma**2))
        elif self.prior_type == "powerlaw":
            alpha = alpha.transpose(1, 2)
            alpha = F.softplus(alpha) + 1e-4
            alpha = alpha.unsqueeze(-1).repeat(1, 1, 1, window_size)
            distances = self.distances.unsqueeze(0).unsqueeze(0).repeat(batch_size, heads, 1, 1).to(alpha.device)
            distances = torch.clamp(distances, min=1.0)
            prior = torch.exp(-alpha * torch.log(distances))
            prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}. Must be 'gaussian' or 'powerlaw'.")

        series = self.dropout(torch.softmax(attn, dim=-1))
        values_out = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            prior_param = sigma if self.prior_type == "gaussian" else alpha
            return values_out.contiguous(), series, prior, prior_param
        return values_out.contiguous(), None, None, None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
        prior_type: str = "gaussian",
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.prior_type = prior_type
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        if prior_type == "powerlaw":
            self.alpha_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        batch_size, length, _ = queries.shape
        _, seq_len, _ = keys.shape
        heads = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(batch_size, length, heads, -1)
        keys = self.key_projection(keys).view(batch_size, seq_len, heads, -1)
        values = self.value_projection(values).view(batch_size, seq_len, heads, -1)
        sigma = self.sigma_projection(x).view(batch_size, length, heads)

        alpha = None
        if self.prior_type == "powerlaw":
            alpha = self.alpha_projection(x).view(batch_size, length, heads)

        out, series, prior, prior_param = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
            alpha=alpha,
        )
        out = out.view(batch_size, length, -1)
        return self.out_projection(out), series, prior, prior_param


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class VanillaAnomalyTransformer(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
        prior_type="gaussian",
    ):
        super().__init__()
        self.output_attention = output_attention
        self.prior_type = prior_type
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size,
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            prior_type=prior_type,
                        ),
                        d_model,
                        n_heads,
                        prior_type=prior_type,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        return enc_out


AnomalyTransformerBaseline = VanillaAnomalyTransformer