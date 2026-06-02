#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.io import save_json
from src.gbm.paths import get_run_dir
from src.gbm.score_variants import SCORE_VARIANTS, apply_score_variants, variant_label


def load_source_scores(source_exp: str) -> pd.DataFrame:
    run_dir = get_run_dir(source_exp)
    path = run_dir / "reports" / "gbm_joint_test_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing GBM test scores: {path}\n"
            f"Run canonical GBM test first (e.g. run_canonical_gbm_attention.bat or test_joint.py)."
        )
    return pd.read_csv(path, parse_dates=["start_date", "end_date"])


def export_variants(source_df: pd.DataFrame, output_exp: str, variants: list[str]) -> dict[str, str]:
    selected = {name: SCORE_VARIANTS[name] for name in variants}
    enriched = apply_score_variants(source_df, selected)
    run_dir = get_run_dir(output_exp)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for variant_name in variants:
        variant_df = enriched.copy()
        variant_df["score"] = variant_df[f"score_{variant_name}"]
        variant_df["score_variant"] = variant_name
        variant_df["score_variant_label"] = variant_label(variant_name)
        variant_df["source_weights"] = json.dumps(selected[variant_name])

        out_path = reports_dir / f"{variant_name}_test_scores.csv"
        variant_df.to_csv(out_path, index=False)
        paths[variant_name] = str(out_path)

        by_ticker_dir = reports_dir / "by_ticker" / variant_name
        by_ticker_dir.mkdir(parents=True, exist_ok=True)
        for ticker, group in variant_df.groupby("ticker"):
            group.to_csv(by_ticker_dir / f"{ticker}_test_scores.csv", index=False)

    combined = enriched.copy()
    for variant_name in variants:
        combined[f"score_{variant_name}"] = enriched[f"score_{variant_name}"]
    combined_path = reports_dir / "all_variants_test_scores.csv"
    combined.to_csv(combined_path, index=False)
    paths["combined"] = str(combined_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive GBM raw score variants (recon / KL / mixed) from component columns without retraining"
    )
    parser.add_argument("--source-exp-name", default="canonical_gbm_attention")
    parser.add_argument("--output-exp-name", default="score_component_ablation")
    parser.add_argument(
        "--variants",
        default="recon_only,association_kl_only,recon_plus_kl,full_default",
        help=f"Comma-separated variant names. Available: {', '.join(SCORE_VARIANTS)}",
    )
    args = parser.parse_args()

    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = sorted(set(variant_names) - set(SCORE_VARIANTS))
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}")

    print(f"[score_ablation] source={args.source_exp_name} -> output={args.output_exp_name}", flush=True)
    source_df = load_source_scores(args.source_exp_name)
    print(f"[score_ablation] loaded {len(source_df)} test windows | tickers={source_df['ticker'].nunique()}", flush=True)

    paths = export_variants(source_df, args.output_exp_name, variant_names)
    manifest_path = get_run_dir(args.output_exp_name) / "score_ablation_manifest.json"
    save_json(
        manifest_path,
        {
            "source_exp_name": args.source_exp_name,
            "output_exp_name": args.output_exp_name,
            "variants": {name: SCORE_VARIANTS[name] for name in variant_names},
            "variant_labels": {name: variant_label(name) for name in variant_names},
            "n_test_windows": int(len(source_df)),
            "ticker_count": int(source_df["ticker"].nunique()),
            "reports": paths,
            "protocol": "Raw scores recomputed from stored test components; no threshold tuning.",
        },
    )
    print(f"Saved {len(variant_names)} variant score files under {get_run_dir(args.output_exp_name) / 'reports'}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
