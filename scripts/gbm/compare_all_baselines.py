#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.baselines.statistical import BASELINE_NAMES
from src.gbm.paths import get_run_dir

NEURAL_BASELINES = ("lstm_autoencoder", "mlp_autoencoder", "cnn1d_lstm_autoencoder")
SKLEARN_BASELINES = ("random_forest", "isolation_forest")
DEFAULT_BASELINES = (
    "rolling_volatility",
    "lstm_autoencoder",
    "cnn1d_lstm_autoencoder",
    "random_forest",
    "isolation_forest",
)


def discover_baselines(baseline_exp: str, requested: list[str] | None) -> list[str]:
    reports_dir = get_run_dir(baseline_exp) / "reports"
    if not reports_dir.exists():
        raise FileNotFoundError(f"Missing baseline reports dir: {reports_dir}")

    available = []
    for path in sorted(reports_dir.glob("*_test_scores.csv")):
        name = path.name.replace("_test_scores.csv", "")
        if name == "baselines_summary":
            continue
        available.append(name)

    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(f"Missing baseline score files: {', '.join(missing)}")
        return requested
    return [name for name in DEFAULT_BASELINES if name in available] or available


def build_series_manifest(gbm_exp: str, baseline_exp: str, baselines: list[str]) -> str:
    parts = [f"gbm={get_run_dir(gbm_exp) / 'reports' / 'gbm_joint_test_scores.csv'}"]
    reports_dir = get_run_dir(baseline_exp) / "reports"
    for baseline in baselines:
        path = reports_dir / f"{baseline}_test_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        parts.append(f"{baseline}={path}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GBM vs multiple baseline raw scores for every ticker")
    parser.add_argument("--gbm-exp-name", default="canonical_gbm_attention")
    parser.add_argument("--baseline-exp-name", default="baseline_comparison")
    parser.add_argument(
        "--baselines",
        default="",
        help=f"Comma-separated baseline names (default: auto). Known: {', '.join(BASELINE_NAMES)}, "
        f"{', '.join(NEURAL_BASELINES)}, {', '.join(SKLEARN_BASELINES)}",
    )
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-true-labels", action="store_true")
    args = parser.parse_args()

    requested = [name.strip() for name in args.baselines.split(",") if name.strip()] or None
    baselines = discover_baselines(args.baseline_exp_name, requested)
    series_manifest = build_series_manifest(args.gbm_exp_name, args.baseline_exp_name, baselines)

    output_dir = get_run_dir(args.baseline_exp_name) / "figures" / "gbm_vs_baselines"
    plot_script = Path(__file__).resolve().parent / "plot_raw_score_comparison.py"

    import subprocess

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
        "GBM vs baselines | ",
        "--all-tickers",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.show_true_labels:
        command.append("--show-true-labels")

    print(f"[compare_all_baselines] baselines={baselines}", flush=True)
    print(f"[compare_all_baselines] output={output_dir}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
