#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


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
            key = key.strip()
            value_text = raw_value.strip()
            if key == "tickers":
                if not value_text or value_text.lower() in {"none", "null", "~", "all"}:
                    config[key] = "all"
                else:
                    config[key] = [item.strip() for item in value_text.split(",") if item.strip()]
            elif key in {"window_size", "score_mav_windows"} and "," in value_text:
                config[key] = [int(item.strip()) for item in value_text.split(",") if item.strip()]
            else:
                config[key] = parse_scalar(value_text)
    return config


def as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in {"true", "yes", "on", "1"}:
            return True
        if lower in {"false", "no", "off", "0"}:
            return False
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a joint financial prior attention pipeline from a flat YAML config")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = load_flat_yaml(config_path)
    root = Path(__file__).resolve().parents[2]

    window_sizes = config.get("window_size", 100)
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    elif isinstance(window_sizes, str):
        window_sizes = [int(window_sizes)]
    elif not isinstance(window_sizes, list):
        window_sizes = [100]

    base_exp_name = str(config.get("exp_name", "experiment3_joint"))
    base_data_path = str(config.get("data_path", "datasets/SP500_event_taxonomy_w100"))

    def resolve_template(value: object, window_size: int, default: str) -> str:
        template = default if value is None else str(value)
        if "{window_size}" in template:
            return template.format(window_size=window_size)
        return template

    def build_command(window_size: int, exp_name: str) -> list[str]:
        command = [
            sys.executable,
            "-u",
            str(root / "scripts" / "gbm" / "run_joint.py"),
        ]

        def add_option(flag: str, value: object) -> None:
            if value is None:
                return
            command.extend([flag, str(value)])

        add_option("--exp-name", exp_name)
        add_option("--data-path", resolve_template(config.get("data_path"), window_size, base_data_path))
        add_option("--device", config.get("device", "auto"))
        add_option("--features", config.get("features", "all"))
        add_option("--split-method", config.get("split_method", "chronological"))
        add_option("--purge-gap", config.get("purge_gap"))
        add_option("--batch-size", config.get("batch_size", 32))
        add_option("--epochs", config.get("epochs", 20))
        add_option("--lr", config.get("lr", 1e-4))
        add_option("--window-size", window_size)
        add_option("--step", config.get("step", 1))
        add_option("--seed", config.get("seed", 42))
        add_option("--d-model", config.get("d_model", 128))
        add_option("--n-heads", config.get("n_heads", 4))
        add_option("--e-layers", config.get("e_layers", 3))
        add_option("--d-ff", config.get("d_ff", 256))
        add_option("--dropout", config.get("dropout", 0.1))
        add_option("--predictive-distribution", config.get("predictive_distribution", "gaussian"))
        add_option("--association-mode", config.get("association_mode", "gaussian_log_return"))
        add_option("--patience", config.get("patience", 5))
        add_option("--dist-weight", config.get("dist_weight", 1.0))
        add_option("--recon-weight", config.get("recon_weight", 1.0))
        add_option("--divergence-weight", config.get("divergence_weight", 0.25))
        add_option("--association-weight", config.get("association_weight", 0.1))
        add_option("--threshold-quantile", config.get("threshold_quantile", 0.95))
        add_option("--threshold-method", config.get("threshold_method", "quantile"))
        add_option("--tolerance-windows", config.get("tolerance_windows", 3))
        add_option("--top-k", config.get("top_k", 10))
        add_option("--spike-percentile", config.get("spike_percentile", 0.95))
        add_option("--tail-days", config.get("tail_days"))
        score_mav_windows = config.get("score_mav_windows")
        if isinstance(score_mav_windows, list):
            command.append("--score-mav-windows")
            command.extend([str(window) for window in score_mav_windows])
        else:
            add_option("--score-mav-windows", score_mav_windows)

        if as_bool(config.get("normalize_batch")):
            command.append("--normalize-batch")
        if as_bool(config.get("include_anomalous_train")):
            command.append("--include-anomalous-train")
        if as_bool(config.get("visualize")):
            command.append("--visualize")
        if as_bool(config.get("show_true_labels")):
            command.append("--show-true-labels")
        if as_bool(config.get("full_context")):
            command.append("--full-context")

        tickers = config.get("tickers") or []
        if tickers and tickers != "all":
            command.append("--tickers")
            command.extend([str(ticker) for ticker in tickers])
        return command

    print(f"[run_joint_from_yaml] config={config_path}")

    commands: list[list[str]] = []
    if len(window_sizes) == 1:
        window_size = int(window_sizes[0])
        exp_name = resolve_template(config.get("exp_name"), window_size, base_exp_name)
        commands.append(build_command(window_size, exp_name))
    else:
        for window_size in window_sizes:
            window_size = int(window_size)
            exp_name = resolve_template(config.get("exp_name"), window_size, f"{base_exp_name}_w{window_size}")
            commands.append(build_command(window_size, exp_name))

    for command in commands:
        print("[run_joint_from_yaml] command:")
        print(" ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=root)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
