#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.paths import get_run_dir
from src.gbm.score_variants import SCORE_VARIANTS


def build_series_manifest(output_exp: str, variants: list[str]) -> str:
    reports_dir = get_run_dir(output_exp) / "reports"
    parts = []
    for variant in variants:
        path = reports_dir / f"{variant}_test_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing variant scores: {path}. Run run_score_component_ablation.py first.")
        parts.append(f"{variant}={path}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GBM score component ablation (recon / KL / mixed) per ticker")
    parser.add_argument("--output-exp-name", default="score_component_ablation")
    parser.add_argument(
        "--variants",
        default="recon_only,association_kl_only,recon_plus_kl,full_default",
        help=f"Comma-separated variants. Available: {', '.join(SCORE_VARIANTS)}",
    )
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--all-tickers", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-true-labels", action="store_true")
    args = parser.parse_args()

    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    series_manifest = build_series_manifest(args.output_exp_name, variant_names)
    output_dir = get_run_dir(args.output_exp_name) / "figures" / "score_variants"
    plot_script = Path(__file__).resolve().parent / "plot_raw_score_comparison.py"

    command = [
        sys.executable,
        str(plot_script),
        "--series",
        series_manifest,
        "--data-path",
        args.data_path,
        "--output-dir",
        str(output_dir),
        "--title-prefix",
        "Score ablation | ",
        "--all-tickers",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.show_true_labels:
        command.append("--show-true-labels")

    print(f"[plot_score_ablation] variants={variant_names}", flush=True)
    print(f"[plot_score_ablation] output={output_dir}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
