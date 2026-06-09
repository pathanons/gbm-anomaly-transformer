#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from argparse import Namespace
from pathlib import Path
from typing import Iterable

PIPELINES = {
    "gbm",
    "canonical",
    "log_return",
    "baseline",
    "data_prepare",
    "statistical_baseline",
    "score_ablation",
    "mad_visualize",
    "test",
    "train",
    "validate",
    "visualize",
}

GBM_DEFAULTS = {
    "output_root": None,
    "device": "auto",
    "features": "all",
    "split_method": "chronological",
    "purge_gap": None,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-4,
    "step": 1,
    "seed": 42,
    "d_model": 128,
    "n_heads": 4,
    "e_layers": 3,
    "d_ff": 256,
    "dropout": 0.1,
    "predictive_distribution": "gaussian",
    "association_mode": "gaussian_log_return",
    "loss_mode": "legacy",
    "patience": 5,
    "dist_weight": 1.0,
    "recon_weight": 1.0,
    "divergence_weight": 0.25,
    "association_weight": 0.1,
    "top_k": 10,
    "spike_percentile": 0.95,
    "tail_days": None,
    "score_mode": "legacy",
    "quantile_count": 21,
    "tail_weight_gamma": 2.0,
    "tail_weight_power": 2.0,
    "checkpoint_exp_name": None,
    "tickers": None,
    "normalize_batch": False,
    "include_anomalous_train": False,
    "visualize": False,
    "show_true_labels": False,
    "full_context": False,
    "fast_export": False,
}

STAGE_DEFAULTS = {
    "data_path": "datasets/SP500_event_taxonomy_w100",
    "features": "all",
    "step": 1,
    "seed": 42,
    "threshold_quantile": 0.95,
    "tickers": None,
}


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


def load_flat_yaml(path: Path, list_int_keys: Iterable[str] = ()) -> dict[str, object]:
    list_int_key_set = set(list_int_keys)
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
            elif key in list_int_key_set and "," in value_text:
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


def expand_window_sizes(value: object, default: int = 100) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(value)]
    if isinstance(value, list):
        return [int(item) for item in value]
    return [default]


def resolve_template(value: object, window_size: int, default: str) -> str:
    template = default if value is None else str(value)
    if "{window_size}" in template:
        return template.format(window_size=window_size)
    return template


def pipeline_name(config: dict[str, object]) -> str:
    pipeline = str(config.get("pipeline", "gbm")).lower()
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {', '.join(sorted(PIPELINES))}; got {pipeline!r}")
    return pipeline


def normalize_common(values: dict[str, object]) -> dict[str, object]:
    if values.get("tickers") == "all":
        values["tickers"] = None
    values["normalize_batch"] = as_bool(values.get("normalize_batch"))
    values["include_anomalous_train"] = as_bool(values.get("include_anomalous_train"))
    values["visualize"] = as_bool(values.get("visualize"))
    values["show_true_labels"] = as_bool(values.get("show_true_labels"))
    values["full_context"] = as_bool(values.get("full_context"))
    values["fast_export"] = as_bool(values.get("fast_export"))
    return values


def namespace_for_gbm(config: dict[str, object], window_size: int, exp_name: str) -> Namespace:
    pipeline = pipeline_name(config)
    values = {**GBM_DEFAULTS, **config}
    values["exp_name"] = exp_name
    values["data_path"] = resolve_template(
        config.get("data_path"),
        window_size,
        str(config.get("data_path", "datasets/SP500_event_taxonomy_w100")),
    )
    values["window_size"] = window_size
    if pipeline == "canonical" and "association_mode" not in config:
        values["association_mode"] = "canonical_gbm"
    if pipeline == "log_return" and "association_mode" not in config:
        values["association_mode"] = "gaussian_log_return"
    return Namespace(**normalize_common(values))


def namespace_for_stage(config: dict[str, object], window_size: int | None = None, exp_name: str | None = None) -> Namespace:
    values = {**STAGE_DEFAULTS, **config}
    if exp_name is not None:
        values["exp_name"] = exp_name
    if window_size is not None:
        values["window_size"] = window_size
        values["data_path"] = resolve_template(values.get("data_path"), window_size, str(values["data_path"]))
    return Namespace(**normalize_common(values))


def run_gbm_config(config: dict[str, object], window_size: int, exp_name: str, dry_run: bool) -> None:
    args = namespace_for_gbm(config, window_size, exp_name)
    print("[main] internal pipeline: train -> validate -> test")
    print(f"[main] exp={args.exp_name} data={args.data_path} model={args.association_mode} device={args.device}")
    if dry_run:
        return
    from src.gbm.test import test_model, validate_model
    from src.gbm.train import train_model

    train_model(args)
    validate_model(args)
    test_model(args)


def run_data_prepare(config: dict[str, object], window_size: int, exp_name: str, dry_run: bool) -> None:
    args = namespace_for_gbm(config, window_size, exp_name)
    print(f"[main] internal pipeline: data preparation exp={args.exp_name} data={args.data_path}")
    if dry_run:
        return
    from src.gbm.datasets import create_joint_manifest, discover_tickers, get_run_dir, save_joint_manifest

    tickers = args.tickers if args.tickers else discover_tickers(args.data_path)
    manifest, window_store, scaler = create_joint_manifest(
        data_path=args.data_path,
        tickers=tickers,
        window_size=args.window_size,
        step=args.step,
        features=args.features,
        seed=args.seed,
        split_method=args.split_method,
        purge_gap=args.purge_gap,
        train_normal_only=not args.include_anomalous_train,
    )
    run_dir = get_run_dir(args.exp_name, args.output_root)
    manifest_path = save_joint_manifest(
        run_dir,
        manifest,
        args.seed,
        args.window_size,
        args.step,
        args.features,
        split_method=args.split_method,
        purge_gap=args.purge_gap,
        train_normal_only=not args.include_anomalous_train,
    )
    print(f"[main] prepared windows={len(manifest)} manifest={manifest_path}")


def run_config(config_path: Path, dry_run: bool) -> None:
    config = load_flat_yaml(config_path, list_int_keys={"window_size", "score_mav_windows"})
    pipeline = pipeline_name(config)
    window_sizes = expand_window_sizes(config.get("window_size", 100), default=100)
    base_exp_name = str(config.get("exp_name", "experiment"))

    print(f"[main] config={config_path}")
    if pipeline in {"gbm", "canonical", "log_return"}:
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            run_gbm_config(config, window_size, exp_name, dry_run)
        return

    if pipeline == "data_prepare":
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            run_data_prepare(config, window_size, exp_name, dry_run)
        return

    if pipeline in {"train", "validate", "test"}:
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_gbm(config, window_size, exp_name)
            print(f"[main] internal pipeline: {pipeline} exp={args.exp_name} data={args.data_path}")
            if dry_run:
                continue
            if pipeline == "train":
                from src.gbm.train import train_model

                train_model(args)
            elif pipeline == "validate":
                from src.gbm.test import validate_model

                validate_model(args)
            else:
                from src.gbm.test import test_model

                test_model(args)
        return

    if pipeline in {"baseline", "statistical_baseline"}:
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_stage(config, window_size, exp_name)
            print(f"[main] internal pipeline: statistical baselines exp={args.exp_name} data={args.data_path}")
            if dry_run:
                continue
            from src.gbm.statistics import run_statistical_baselines

            run_statistical_baselines(args)
        return

    if pipeline == "score_ablation":
        args = namespace_for_stage(config)
        print(f"[main] internal pipeline: score ablation source={args.source_exp_name} output={args.output_exp_name}")
        if dry_run:
            return
        from src.gbm.statistics import run_score_ablation

        run_score_ablation(args)
        return

    if pipeline in {"mad_visualize", "visualize"}:
        args = namespace_for_stage(config)
        print(f"[main] internal pipeline: mad visualize csv={args.csv} out={args.out}")
        if dry_run:
            return
        from src.gbm.visualize import run_mad_visualize

        run_mad_visualize(args)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Single YAML-driven entry point for this repo")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved work without running")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    run_config(config_path, args.dry_run)


if __name__ == "__main__":
    main()
