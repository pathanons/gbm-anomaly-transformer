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
