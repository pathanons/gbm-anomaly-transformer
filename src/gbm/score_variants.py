from __future__ import annotations

from typing import Dict

import pandas as pd

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
