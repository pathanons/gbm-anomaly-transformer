from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.gbm.model import AnomalyTransformer, gaussian_transition_timestamp_posterior, temporal_prior
from src.gbm.score import (
    apply_score_variants,
    binary_metrics,
    compute_variant_score,
    gaussian_nll,
    predictive_nll,
    score_windows,
)


def test_transition_posterior_is_causal_and_row_normalized() -> None:
    returns = torch.tensor([[0.0, 0.01, -0.02, 0.03]], dtype=torch.float32)
    drift = torch.zeros(1, 2, 4)
    sigma = torch.ones(1, 2, 4) * 0.2

    posterior = gaussian_transition_timestamp_posterior(returns, drift, sigma, n_heads=2)

    assert posterior.shape == (1, 2, 4, 4)
    assert torch.allclose(posterior.sum(dim=-1), torch.ones(1, 2, 4), atol=1e-5)
    assert torch.allclose(posterior[:, :, 0, :], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.all(posterior[:, :, 2, 3] < 1e-6)


def test_model_forward_shapes_for_none_association() -> None:
    model = AnomalyTransformer(
        win_size=8,
        enc_in=4,
        c_out=4,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
        dropout=0.0,
        association_mode="none",
    )
    x = torch.randn(3, 8, 4)
    returns = torch.randn(3, 8)
    time_deltas = torch.ones(3, 8)

    recon, mu, sigma, nu, attn_maps, obs_mu, obs_sigma, hidden, association = model(
        x,
        returns=returns,
        time_deltas=time_deltas,
        return_attention=True,
    )

    assert recon.shape == x.shape
    assert mu.shape == (3,)
    assert sigma.shape == (3,)
    assert nu is None
    assert hidden.shape == (3, 8, 8)
    assert association.shape == (3,)
    assert len(attn_maps) == 1


def test_temporal_prior_is_causal() -> None:
    prior = temporal_prior(batch_size=1, n_heads=1, length=5, device=torch.device("cpu"))

    assert prior.shape == (1, 1, 5, 5)
    assert torch.allclose(prior.sum(dim=-1), torch.ones(1, 1, 5), atol=1e-6)
    assert torch.all(prior[0, 0].triu(diagonal=1) == 0)


def test_score_components_and_variants() -> None:
    recon = torch.tensor([1.0, 2.0])
    nll = torch.tensor([10.0, 20.0])
    divergence = torch.tensor([0.5, 1.0])
    association = torch.tensor([3.0, 4.0])

    score = score_windows(
        recon,
        nll,
        divergence,
        association=association,
        dist_weight=1.0,
        recon_weight=0.0,
        divergence_weight=0.0,
        association_weight=0.1,
    )

    assert torch.allclose(score, torch.tensor([10.3, 20.4]))

    frame = pd.DataFrame(
        {
            "reconstruction_error": [1.0],
            "nll": [2.0],
            "divergence": [4.0],
            "association_discrepancy": [8.0],
        }
    )
    assert compute_variant_score(frame, {"dist_weight": 1, "recon_weight": 1, "divergence_weight": 0, "association_weight": 0}).iloc[0] == 3.0
    assert "score_nll_only" in apply_score_variants(frame, {"nll_only": {"dist_weight": 1, "recon_weight": 0, "divergence_weight": 0, "association_weight": 0}}).columns


def test_predictive_nll_validates_distribution() -> None:
    returns = torch.zeros(2, 4)
    mu = torch.zeros(2)
    sigma = torch.ones(2)

    nll = gaussian_nll(returns, mu, sigma)

    assert nll.shape == (2, 4)
    assert torch.isfinite(nll).all()
    with pytest.raises(ValueError):
        predictive_nll(returns, mu, sigma, distribution="bad")


def test_binary_metrics_handles_regular_and_single_class_cases() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)

    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["roc_auc"] == 1.0

    single = binary_metrics([1, 1], [0.2, 0.8], threshold=0.5)
    assert math.isnan(single["roc_auc"])
