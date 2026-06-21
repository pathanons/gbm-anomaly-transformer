#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
from argparse import Namespace
from pathlib import Path
from typing import Iterable

PIPELINES = {
    "gbm",
    "canonical",
    "log_return",
    "baseline",
    "data_prepare",
    "stock_feature_prepare",
    "statistical_baseline",
    "score_ablation",
    "mad_visualize",
    "test",
    "train",
    "validate",
    "visualize",
    "attention_visualize",
    "model_diagnostic_panels",
    "endpoint_logreturn_score_panels",
    "nextday_report_figures",
    "dataset_difference_histograms",
    "dataset_logreturn_histograms",
    "experiment_suite",
}

GBM_DEFAULTS = {
    "output_root": None,
    "device": "auto",
    "features": "all",
    "target": "window_returns",
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
    "loss_combine_mode": "weighted_sum",
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
    "outlier_k": 9.0,
    "score_formula": "sum",
    "mad_out": None,
    "threshold_mode": "delta",
    "show_true_labels": False,
    "full_context": False,
    "fast_export": False,
    "attention_split": "test",
    "attention_top_k": 5,
    "attention_per_ticker": None,
    "attention_select_by": "association_discrepancy",
    "attention_score_csv": None,
    "attention_out": None,
    "attention_layer": 0,
    "attention_head": 0,
    "attention_mad_k": 10.0,
    "attention_score_formula": "sum",
    "attention_overlap_only": False,
    "attention_mask_diagonal": False,
    "price_dir": "datasets/SP500",
    "price_z_thr": 3.0,
    "attention_make_plots": True,
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


def parse_csv_values(value: object, cast=str) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def float_tag(value: object) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


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


def resolve_suite_template(value: object, params: dict[str, object], default: str) -> str:
    template = default if value is None else str(value)
    return template.format(**params)


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
    values["attention_make_plots"] = as_bool(values.get("attention_make_plots"), default=True)
    values["attention_overlap_only"] = as_bool(values.get("attention_overlap_only"), default=False)
    values["attention_mask_diagonal"] = as_bool(values.get("attention_mask_diagonal"), default=False)
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
    if getattr(args, "visualize", False):
        from src.gbm.datasets import get_run_dir
        from src.gbm.visualize import run_mad_visualize

        k_tag = str(float(getattr(args, "outlier_k", 9.0))).rstrip("0").rstrip(".").replace(".", "p")
        run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
        score_csv = run_dir / "reports" / "test_scores.csv"
        out_dir = Path(str(getattr(args, "mad_out", "") or (run_dir / "figures" / f"mad_k{k_tag}")))
        viz_args = Namespace(
            csv=str(score_csv),
            out=str(out_dir),
            tickers=getattr(args, "tickers", None),
            label_names=getattr(args, "label_names", None),
            price_dir=getattr(args, "price_dir", "datasets/SP500"),
            price_z_thr=getattr(args, "price_z_thr", 3.0),
            outlier_k=getattr(args, "outlier_k", 9.0),
            score_formula=getattr(args, "score_formula", "sum"),
            threshold_mode=getattr(args, "threshold_mode", "delta"),
        )
        run_mad_visualize(viz_args)


def suite_trials(config: dict[str, object]) -> list[dict[str, object]]:
    heads = parse_csv_values(config.get("n_heads_values", config.get("n_heads", 4)), int)
    layers = parse_csv_values(config.get("e_layers_values", config.get("e_layers", 3)), int)
    learning_rates = parse_csv_values(config.get("lr_values", config.get("lr", 1e-4)), float)
    epochs = parse_csv_values(config.get("epoch_values", config.get("epochs", 20)), int)
    seeds = parse_csv_values(config.get("seed_values", config.get("seed", 42)), int)
    if not all([heads, layers, learning_rates, epochs, seeds]):
        raise ValueError("experiment_suite requires non-empty heads/layers/lr/epoch/seed values")

    trials = []
    for n_heads, e_layers, lr, epoch_count, seed in itertools.product(heads, layers, learning_rates, epochs, seeds):
        trials.append(
            {
                "n_heads": n_heads,
                "e_layers": e_layers,
                "lr": lr,
                "epochs": epoch_count,
                "seed": seed,
            }
        )
    return trials


def run_experiment_suite(config: dict[str, object], dry_run: bool) -> None:
    trials = suite_trials(config)
    max_trials = config.get("max_trials", None)
    if max_trials is not None:
        trials = trials[: int(max_trials)]

    mad_k_values = parse_csv_values(config.get("mad_k_values", "9"), float)
    window_size = int(config.get("window_size", 100))
    base_exp_name = str(config.get("exp_name", "best_nllassoc_suite"))
    output_root = config.get("output_root")
    print(
        f"[main] experiment suite: trials={len(trials)} window_size={window_size} "
        f"mad_k_values={','.join(str(k) for k in mad_k_values)}"
    )

    for idx, trial in enumerate(trials, start=1):
        params = {
            **trial,
            "trial": idx,
            "window_size": window_size,
            "lr_tag": float_tag(trial["lr"]),
        }
        default_name = (
            f"{base_exp_name}_t{idx}_w{window_size}_h{trial['n_heads']}"
            f"_l{trial['e_layers']}_lr{params['lr_tag']}_ep{trial['epochs']}_s{trial['seed']}"
        )
        exp_name = resolve_suite_template(config.get("exp_name_template"), params, default_name)
        trial_config = {
            **config,
            **trial,
            "pipeline": str(config.get("model_pipeline", "log_return")),
            "exp_name": exp_name,
            "window_size": window_size,
        }
        print(f"[main] suite trial {idx}/{len(trials)} exp={exp_name}")
        run_gbm_config(trial_config, window_size, exp_name, dry_run)

        if not mad_k_values:
            continue
        score_csv = str(
            config.get(
                "score_csv_template",
                "D:/AnomalyTransformerRuns/experiments/{exp_name}/reports/test_scores.csv",
            )
        ).format(exp_name=exp_name)
        for mad_k in mad_k_values:
            k_tag = str(mad_k).replace(".", "p")
            out_dir = str(
                config.get(
                    "mad_out_template",
                    "D:/AnomalyTransformerRuns/experiments/{exp_name}/figures/mad_k{k_tag}",
                )
            ).format(exp_name=exp_name, k=mad_k, k_tag=k_tag)
            print(f"[main] suite threshold k={mad_k:g} csv={score_csv} out={out_dir}")
            if dry_run:
                continue
            args = namespace_for_stage(
                {
                    **config,
                    "pipeline": "mad_visualize",
                    "csv": score_csv,
                    "out": out_dir,
                    "outlier_k": mad_k,
                }
            )
            from src.gbm.visualize import run_mad_visualize

            run_mad_visualize(args)


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
        target=getattr(args, "target", "window_returns"),
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
        target=getattr(args, "target", "window_returns"),
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

    if pipeline == "stock_feature_prepare":
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_stage(config, window_size, exp_name)
            if "prepared_data_path" in config:
                args.prepared_data_path = resolve_template(config.get("prepared_data_path"), window_size, str(config["prepared_data_path"]))
            print(
                "[main] internal pipeline: stock feature prepare "
                f"source={getattr(args, 'source_data_path', 'datasets/SP500')} out={args.prepared_data_path}"
            )
            if dry_run:
                continue
            from src.gbm.datasets import prepare_stock_feature_dataset

            prepare_stock_feature_dataset(args)
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

    if pipeline == "dataset_difference_histograms":
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_stage(config, window_size, exp_name)
            print(
                "[main] internal pipeline: dataset difference histograms "
                f"data={args.data_path} window_size={args.window_size}"
            )
            if dry_run:
                continue
            from src.gbm.visualize import run_dataset_difference_histograms

            run_dataset_difference_histograms(args)
        return

    if pipeline == "dataset_logreturn_histograms":
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_stage(config, window_size, exp_name)
            if "histogram_all_out" in config:
                args.histogram_all_out = resolve_template(config.get("histogram_all_out"), window_size, str(config["histogram_all_out"]))
            if "histogram_anomaly_out" in config:
                args.histogram_anomaly_out = resolve_template(config.get("histogram_anomaly_out"), window_size, str(config["histogram_anomaly_out"]))
            print(
                "[main] internal pipeline: dataset log-return histograms "
                f"data={args.data_path} window_size={args.window_size}"
            )
            if dry_run:
                continue
            from src.gbm.visualize import run_dataset_logreturn_histograms

            run_dataset_logreturn_histograms(args)
        return

    if pipeline == "attention_visualize":
        for window_size in window_sizes:
            default_name = f"{base_exp_name}_w{window_size}" if len(window_sizes) > 1 else base_exp_name
            exp_name = resolve_template(config.get("exp_name"), window_size, default_name)
            args = namespace_for_gbm(config, window_size, exp_name)
            print(
                f"[main] internal pipeline: attention visualize exp={args.exp_name} "
                f"split={args.attention_split} top_k={args.attention_top_k}"
            )
            if dry_run:
                continue
            from src.gbm.test import export_attention_artifacts

            export_attention_artifacts(args)
        return

    if pipeline == "model_diagnostic_panels":
        args = namespace_for_stage(config)
        print(f"[main] internal pipeline: model diagnostic panels csv={args.csv} out={args.out}")
        if dry_run:
            return
        from src.gbm.visualize import run_model_diagnostic_panels

        run_model_diagnostic_panels(args)
        return

    if pipeline == "endpoint_logreturn_score_panels":
        args = namespace_for_stage(config)
        print(f"[main] internal pipeline: endpoint log-return score panels csv={args.csv} out={args.out}")
        if dry_run:
            return
        from src.gbm.visualize import run_endpoint_logreturn_score_panels

        run_endpoint_logreturn_score_panels(args)
        return

    if pipeline == "nextday_report_figures":
        args = namespace_for_stage(config)
        print(f"[main] internal pipeline: next-day report figures out={args.out}")
        if dry_run:
            return
        from src.gbm.visualize import run_nextday_report_figures

        run_nextday_report_figures(args)
        return

    if pipeline == "experiment_suite":
        run_experiment_suite(config, dry_run)
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
