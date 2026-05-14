from __future__ import annotations

import math

import torch


def gaussian_nll(returns: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(sigma, min=1e-4)
    mu = mu.unsqueeze(-1)
    sigma = sigma.unsqueeze(-1)
    return 0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma) + ((returns - mu) ** 2) / (sigma ** 2))


def gaussian_wasserstein(mu_a: torch.Tensor, sigma_a: torch.Tensor, mu_b: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
    sigma_a = torch.clamp(sigma_a, min=1e-4)
    sigma_b = torch.clamp(sigma_b, min=1e-4)
    return (mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2


def score_windows(recon_error: torch.Tensor, nll: torch.Tensor, divergence: torch.Tensor, recon_weight: float = 0.25) -> torch.Tensor:
    return nll + divergence + recon_weight * recon_error

