#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.gbm.paths import get_run_dir


def parse_csv(raw: object) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scalar(raw: str):
    value = raw.strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null", "~", ""}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_flat_yaml(path: Path) -> dict[str, object]:
    config: dict[str, object] = {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid config line: {line.rstrip()}")
            key, raw_value = line.split(":", 1)
            config[key.strip()] = parse_scalar(raw_value)
    return config


def config_default(config: dict[str, object], key: str, fallback):
    return config.get(key, fallback)


def safe_token(value: str) -> str:
    return value.replace(".", "p").replace("-", "m").replace(" ", "_")


def python_has_torch(python: Path | str) -> bool:
    try:
        completed = subprocess.run(
            [str(python), "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return completed.returncode == 0
    except OSError:
        return False


def resolve_python(requested: str | None, root: Path) -> str:
    if requested:
        return requested
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        Path(r"C:\Users\Acer\anaconda3\python.exe"),
        Path(r"C:\Users\Acer\miniconda3\python.exe"),
        Path(r"C:\Users\Acer\anaconda3\envs\anomaly-transformer-exp1\python.exe"),
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists() and python_has_torch(candidate):
            return str(candidate)
    return sys.executable


def run_command(command: list[str], root: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=root)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_eval_common(args: argparse.Namespace) -> list[str]:
    common = [
        "--checkpoint-exp-name",
        args.checkpoint_exp_name,
        "--data-path",
        args.data_path,
        "--device",
        args.device,
        "--features",
        args.features,
        "--split-method",
        args.split_method,
        "--batch-size",
        str(args.batch_size),
        "--window-size",
        str(args.window_size),
        "--step",
        str(args.step),
        "--seed",
        str(args.seed),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--e-layers",
        str(args.e_layers),
        "--d-ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--predictive-distribution",
        args.predictive_distribution,
        "--association-mode",
        args.association_mode,
        "--dist-weight",
        str(args.dist_weight),
        "--recon-weight",
        str(args.recon_weight),
        "--divergence-weight",
        str(args.divergence_weight),
        "--association-weight",
        str(args.association_weight),
    ]
    if args.purge_gap is not None:
        common.extend(["--purge-gap", str(args.purge_gap)])
    if args.include_anomalous_train:
        common.append("--include-anomalous-train")
    if args.normalize_batch:
        common.append("--normalize-batch")
    if args.tickers:
        common.append("--tickers")
        common.extend(args.tickers)
    return common


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Optional flat YAML config file")
    pre_args, remaining = pre_parser.parse_known_args()
    config = load_flat_yaml(Path(pre_args.config)) if pre_args.config else {}

    parser = argparse.ArgumentParser(
        description=(
            "Run validation/test decision sweeps from one trained joint checkpoint. "
            "This does not retrain the model."
        ),
        parents=[pre_parser],
    )
    parser.add_argument("--checkpoint-exp-name", default=config_default(config, "checkpoint_exp_name", None), required="checkpoint_exp_name" not in config, help="Experiment containing models/gbm_joint.pt")
    parser.add_argument("--sweep-name", default=config_default(config, "sweep_name", None), help="Name prefix for generated decision-sweep experiments")
    parser.add_argument("--data-path", default=config_default(config, "data_path", "datasets/SP500_event_taxonomy_w100"))
    parser.add_argument("--tickers", nargs="*", default=config_default(config, "tickers", None))
    parser.add_argument("--device", default=config_default(config, "device", "auto"))
    parser.add_argument("--features", default=config_default(config, "features", "all"), choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true", default=bool(config_default(config, "normalize_batch", False)))
    parser.add_argument("--split-method", default=config_default(config, "split_method", "chronological"), choices=["chronological", "random"])
    parser.add_argument("--purge-gap", type=int, default=config_default(config, "purge_gap", None))
    parser.add_argument("--include-anomalous-train", action="store_true", default=bool(config_default(config, "include_anomalous_train", False)))
    parser.add_argument("--batch-size", type=int, default=config_default(config, "batch_size", 32))
    parser.add_argument("--window-size", type=int, default=config_default(config, "window_size", 100))
    parser.add_argument("--step", type=int, default=config_default(config, "step", 1))
    parser.add_argument("--seed", type=int, default=config_default(config, "seed", 42))
    parser.add_argument("--d-model", type=int, default=config_default(config, "d_model", 128))
    parser.add_argument("--n-heads", type=int, default=config_default(config, "n_heads", 4))
    parser.add_argument("--e-layers", type=int, default=config_default(config, "e_layers", 3))
    parser.add_argument("--d-ff", type=int, default=config_default(config, "d_ff", 256))
    parser.add_argument("--dropout", type=float, default=config_default(config, "dropout", 0.1))
    parser.add_argument("--predictive-distribution", default=config_default(config, "predictive_distribution", "student_t"), choices=["gaussian", "student_t"])
    parser.add_argument(
        "--association-mode",
        default=config_default(config, "association_mode", "canonical_gbm"),
        choices=["gaussian_log_return", "canonical_gbm", "temporal", "none"],
    )
    parser.add_argument("--dist-weight", type=float, default=config_default(config, "dist_weight", 1.0))
    parser.add_argument("--recon-weight", type=float, default=config_default(config, "recon_weight", 1.0))
    parser.add_argument("--divergence-weight", type=float, default=config_default(config, "divergence_weight", 0.25))
    parser.add_argument("--association-weight", type=float, default=config_default(config, "association_weight", 0.1))
    parser.add_argument(
        "--threshold-methods",
        default=config_default(config, "threshold_methods", "conformal,per_ticker_conformal,evt,tail_probability,var"),
        help="Comma-separated methods to sweep",
    )
    parser.add_argument("--threshold-quantiles", default=config_default(config, "threshold_quantiles", "0.90,0.95,0.975,0.99"))
    parser.add_argument("--evt-tail-quantiles", default=config_default(config, "evt_tail_quantiles", "0.85,0.90,0.95"))
    parser.add_argument("--tolerance-windows", type=int, default=config_default(config, "tolerance_windows", 3))
    parser.add_argument("--output-root", default=config_default(config, "output_root", None))
    parser.add_argument("--python", default=config_default(config, "python", None), help="Python executable with torch installed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(remaining)

    if isinstance(args.tickers, str):
        if args.tickers.lower() in {"all", "none", "null", "~", ""}:
            args.tickers = None
        else:
            args.tickers = parse_csv(args.tickers)

    if args.output_root:
        os.environ["AT_OUTPUT_ROOT"] = args.output_root

    root = Path(__file__).resolve().parents[2]
    python_executable = resolve_python(args.python, root)
    if not python_has_torch(python_executable):
        raise RuntimeError(
            f"Selected Python does not have torch installed: {python_executable}\n"
            "Pass --python C:\\Users\\Acer\\anaconda3\\python.exe or activate the conda environment before running."
        )
    checkpoint_run_dir = get_run_dir(args.checkpoint_exp_name, args.output_root)
    checkpoint_path = checkpoint_run_dir / "models" / "gbm_joint.pt"
    if not args.dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing source checkpoint: {checkpoint_path}\n"
            "Set --output-root or AT_OUTPUT_ROOT to the root containing experiments/. "
            "For the Windows review runs, use --output-root D:\\AnomalyTransformerRuns."
        )

    methods = parse_csv(args.threshold_methods)
    threshold_quantiles = parse_csv(args.threshold_quantiles)
    evt_tail_quantiles = parse_csv(args.evt_tail_quantiles)
    sweep_prefix = args.sweep_name or f"{args.checkpoint_exp_name}_decision_sweep"
    eval_common = build_eval_common(args)

    rows: list[dict[str, object]] = []
    for method in methods:
        tail_values = evt_tail_quantiles if method == "evt" else ["0.90"]
        for threshold_quantile in threshold_quantiles:
            for evt_tail_quantile in tail_values:
                exp_name = f"{sweep_prefix}_{method}_q{safe_token(threshold_quantile)}"
                if method == "evt":
                    exp_name += f"_tail{safe_token(evt_tail_quantile)}"

                validate_cmd = [
                    python_executable,
                    "-u",
                    str(root / "scripts" / "gbm" / "validate_joint.py"),
                    "--exp-name",
                    exp_name,
                    *eval_common,
                    "--threshold-method",
                    method,
                    "--threshold-quantile",
                    threshold_quantile,
                    "--evt-tail-quantile",
                    evt_tail_quantile,
                ]
                test_cmd = [
                    python_executable,
                    "-u",
                    str(root / "scripts" / "gbm" / "test_joint.py"),
                    "--exp-name",
                    exp_name,
                    *eval_common,
                    "--tolerance-windows",
                    str(args.tolerance_windows),
                ]

                print(f"=== decision sweep: {exp_name} ===", flush=True)
                run_command(validate_cmd, root, args.dry_run)
                run_command(test_cmd, root, args.dry_run)
                if args.dry_run:
                    continue

                metrics_path = get_run_dir(exp_name, args.output_root) / "reports" / "gbm_joint_metrics.json"
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                rows.append(
                    {
                        "exp_name": exp_name,
                        "checkpoint_exp_name": args.checkpoint_exp_name,
                        "threshold_method": method,
                        "threshold_quantile": float(threshold_quantile),
                        "evt_tail_quantile": float(evt_tail_quantile) if method == "evt" else None,
                        "association_mode": args.association_mode,
                        "features": args.features,
                        "predictive_distribution": args.predictive_distribution,
                        "precision": metrics.get("precision"),
                        "sensitivity": metrics.get("sensitivity"),
                        "specificity": metrics.get("specificity"),
                        "f1_score": metrics.get("f1_score"),
                        "roc_auc": metrics.get("roc_auc"),
                        "pr_auc": metrics.get("pr_auc"),
                        "tp": metrics.get("tp"),
                        "fp": metrics.get("fp"),
                        "tn": metrics.get("tn"),
                        "fn": metrics.get("fn"),
                        "threshold": metrics.get("threshold"),
                        "threshold_score_column": metrics.get("threshold_score_column"),
                        "decision_rule": metrics.get("decision_rule"),
                    }
                )

    if args.dry_run:
        return

    summary = pd.DataFrame(rows)
    sweep_dir = get_run_dir(sweep_prefix, args.output_root) / "reports"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "decision_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.sort_values("f1_score", ascending=False).to_string(index=False))
    print(f"Saved decision sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
