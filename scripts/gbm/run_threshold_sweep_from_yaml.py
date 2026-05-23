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
            if key in {"window_size", "quantiles"} and "," in value_text:
                config[key] = [item.strip() for item in value_text.split(",") if item.strip()]
            else:
                config[key] = parse_scalar(value_text)
    return config


def parse_quantiles(value: object) -> list[str]:
    if value is None:
        return ["0.00", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09",
                "0.10", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19",
                "0.20", "0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "0.27", "0.28", "0.29",
                "0.30", "0.31", "0.32", "0.33", "0.34", "0.35", "0.36", "0.37", "0.38", "0.39",
                "0.40", "0.41", "0.42", "0.43", "0.44", "0.45", "0.46", "0.47", "0.48", "0.49",
                "0.50", "0.51", "0.52", "0.53", "0.54", "0.55", "0.56", "0.57", "0.58", "0.59",
                "0.60", "0.61", "0.62", "0.63", "0.64", "0.65", "0.66", "0.67", "0.68", "0.69",
                "0.70", "0.71", "0.72", "0.73", "0.74", "0.75", "0.76", "0.77", "0.78", "0.79",
                "0.80", "0.81", "0.82", "0.83", "0.84", "0.85", "0.86", "0.87", "0.88", "0.89",
                "0.90", "0.91", "0.92", "0.93", "0.94", "0.95", "0.96", "0.97", "0.98", "0.99",
                "1.00"]

    if isinstance(value, str):
        if value.lower() == "all":
            return parse_quantiles(None)
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def flatten_windows(value: object) -> list[int]:
    if value is None:
        return [100]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        if "," in value:
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return [int(value)]
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def resolve_template(value: object, window_size: int, default: str) -> str:
    template = default if value is None else str(value)
    if "{window_size}" in template:
        return template.format(window_size=window_size)
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold sweeps from a flat YAML config")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without running them")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config = load_flat_yaml(config_path)
    root = Path(__file__).resolve().parents[2]

    window_sizes = flatten_windows(config.get("window_size", 100))

    quantiles = parse_quantiles(config.get("quantiles", "all"))
    base_exp_name = str(config.get("exp_name", "experiment3_joint"))
    base_label = str(config.get("label", ""))

    def maybe_add(command: list[str], flag: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, str) and value.lower() in {"", "none", "null", "~"}:
            return
        command.extend([flag, str(value)])

    print(f"[run_threshold_sweep_from_yaml] config={config_path}")

    commands: list[list[str]] = []
    for window_size in window_sizes:
        window_size = int(window_size)
        exp_name = resolve_template(config.get("exp_name"), window_size, f"{base_exp_name}_w{window_size}")
        label_value = resolve_template(config.get("label"), window_size, f"{base_label}_w{window_size}" if base_label else f"{exp_name}")
        command = [
            sys.executable,
            "-u",
            str(root / "scripts" / "gbm" / "sweep_validation_thresholds.py"),
        ]
        if config.get("scores_file"):
            command.extend(["--scores-file", str(config.get("scores_file"))])
        else:
            command.extend(["--exp-name", exp_name])

        maybe_add(command, "--label", label_value if config.get("label") is not None else None)
        maybe_add(command, "--score-column", config.get("score_column", "score"))
        maybe_add(command, "--label-column", config.get("label_column", "y_true"))
        maybe_add(command, "--quantiles", ",".join(quantiles))
        maybe_add(command, "--output-dir", resolve_template(config.get("output_dir"), window_size, "") if config.get("output_dir") else None)
        commands.append(command)

    for command in commands:
        print("[run_threshold_sweep_from_yaml] command:")
        print(" ".join(command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=root)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
