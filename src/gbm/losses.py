from __future__ import annotations

import math

import torch


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
