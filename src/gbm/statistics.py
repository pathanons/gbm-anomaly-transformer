"""Canonical statistics and evaluation surface for GBM experiments."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from src.gbm.datasets import save_json
from src.gbm.score import SCORE_VARIANTS, apply_score_variants, average_precision_score_local, binary_metrics, roc_auc_score_local, variant_label

EPS = 1e-8


def rolling_volatility(returns: np.ndarray) -> float:
    return float(np.std(returns, ddof=0))


def mean_abs_return(returns: np.ndarray) -> float:
    return float(np.mean(np.abs(returns)))


def last_return_zscore(returns: np.ndarray) -> float:
    if len(returns) <= 1:
        return 0.0
    history = returns[:-1]
    return float(abs((returns[-1] - np.mean(history)) / (np.std(history, ddof=0) + EPS)))


def max_return_zscore(returns: np.ndarray) -> float:
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    return float(np.max(np.abs((returns - mean) / (std + EPS))))


def ewma_volatility(returns: np.ndarray, span: float = 20.0) -> float:
    if len(returns) == 0:
        return 0.0
    weights = np.exp(-np.arange(len(returns))[::-1] / max(span, 1.0))
    weights = weights / weights.sum()
    mean = float(np.sum(weights * returns))
    var = float(np.sum(weights * (returns - mean) ** 2))
    return math.sqrt(max(var, 0.0))


def vol_ratio_short_long(returns: np.ndarray, short: int = 5, long: int = 20) -> float:
    if len(returns) < long:
        return float(np.std(returns, ddof=0))
    short_vol = float(np.std(returns[-short:], ddof=0))
    long_vol = float(np.std(returns[-long:], ddof=0))
    return short_vol / (long_vol + EPS)


def cusum_abs_return(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    centered = returns - np.median(returns)
    cumulative = np.cumsum(centered)
    return float(np.max(cumulative) - np.min(cumulative))


STATISTICAL_BASELINES: Dict[str, Callable[[np.ndarray], float]] = {
    "rolling_volatility": rolling_volatility,
    "mean_abs_return": mean_abs_return,
    "last_return_zscore": last_return_zscore,
    "max_return_zscore": max_return_zscore,
    "ewma_volatility": ewma_volatility,
    "vol_ratio_short_long": vol_ratio_short_long,
    "cusum_abs_return": cusum_abs_return,
}
BASELINE_NAMES = sorted(STATISTICAL_BASELINES)


def score_manifest(
    manifest: pd.DataFrame,
    window_store: dict,
    baseline_name: str,
    scorer: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        ticker_store = window_store[row.ticker]
        span = int(row.end_idx - row.start_idx)
        returns = ticker_store["returns"][int(row.start_idx) : int(row.start_idx) + span]
        score = scorer(np.nan_to_num(returns.astype(float)))
        if not math.isfinite(score):
            score = 0.0
        rows.append(
            {
                "baseline": baseline_name,
                "sample_id": int(row.sample_id),
                "ticker": row.ticker,
                "split": row.split,
                "start_idx": int(row.start_idx),
                "end_idx": int(row.end_idx),
                "start_date": row.start_date,
                "end_date": row.end_date,
                "y_true": int(row.y_true),
                "score": float(score),
            }
        )
    return pd.DataFrame(rows)


def parse_baselines(raw: str | None) -> list[str]:
    if raw is None:
        return BASELINE_NAMES[:4]
    names = [name.strip() for name in str(raw).split(",") if name.strip()]
    if "all" in names:
        return BASELINE_NAMES.copy()
    unknown = sorted(set(names) - set(STATISTICAL_BASELINES))
    if unknown:
        raise ValueError(f"Unknown baseline(s): {', '.join(unknown)}. Available: {', '.join(BASELINE_NAMES)}")
    return names


def summarize_baseline(scores: pd.DataFrame, threshold_quantile: float) -> dict[str, float | int | str]:
    val_scores = scores.loc[scores["split"] == "val", "score"].to_numpy(dtype=float)
    test_scores = scores.loc[scores["split"] == "test"].copy()
    threshold = float(pd.Series(val_scores).quantile(threshold_quantile))
    metrics = binary_metrics(test_scores["y_true"].tolist(), test_scores["score"].tolist(), threshold)
    metrics.update(
        {
            "baseline": str(scores["baseline"].iloc[0]),
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
            "val_windows": int(len(val_scores)),
            "test_windows": int(len(test_scores)),
            "ticker_count": int(test_scores["ticker"].nunique()),
            "anomaly_rate": float(test_scores["y_true"].mean()),
            "mean_score": float(test_scores["score"].mean()),
            "std_score": float(test_scores["score"].std(ddof=0)),
        }
    )
    return metrics


def run_statistical_baselines(args) -> pd.DataFrame:
    from src.gbm.datasets import create_joint_manifest, discover_tickers, get_run_dir

    tickers = args.tickers if args.tickers else discover_tickers(args.data_path)
    if not tickers:
        raise SystemExit(f"No tickers found in {args.data_path}")

    selected_baselines = parse_baselines(getattr(args, "baselines", None))
    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest, window_store, _ = create_joint_manifest(
        data_path=args.data_path,
        tickers=tickers,
        window_size=args.window_size,
        step=args.step,
        features=args.features,
        seed=args.seed,
    )

    summary_rows = []
    score_paths = {}
    threshold_quantile = float(getattr(args, "threshold_quantile", 0.95))
    for baseline_name in selected_baselines:
        print(f"[statistics] scoring {baseline_name}", flush=True)
        scores = score_manifest(manifest, window_store, baseline_name, STATISTICAL_BASELINES[baseline_name])
        metrics = summarize_baseline(scores, threshold_quantile)
        summary_rows.append(metrics)
        score_path = reports_dir / f"{baseline_name}_scores.csv"
        scores.to_csv(score_path, index=False)
        score_paths[baseline_name] = str(score_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "statistical_baselines_summary.csv"
    metrics_path = reports_dir / "statistical_baselines_metrics.json"
    manifest_path = run_dir / "statistical_baselines_manifest.json"
    summary.to_csv(summary_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    save_json(
        manifest_path,
        {
            "exp_name": args.exp_name,
            "data_path": args.data_path,
            "tickers": list(tickers),
            "seed": args.seed,
            "window_size": args.window_size,
            "step": args.step,
            "features": args.features,
            "threshold_quantile": threshold_quantile,
            "split_counts": manifest["split"].value_counts().to_dict(),
            "reports": {"summary": str(summary_path), "metrics": str(metrics_path), "scores": score_paths},
            "selection_protocol": "threshold selected from validation scores only; test used for final scoring only",
        },
    )
    print(summary.to_string(index=False))
    print(f"Saved summary to {summary_path}")
    return summary


def load_source_scores(source_exp: str) -> pd.DataFrame:
    from src.gbm.datasets import get_run_dir

    run_dir = get_run_dir(source_exp)
    candidates = [
        run_dir / "reports" / "test_scores.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, parse_dates=["start_date", "end_date"])
    raise FileNotFoundError(f"Missing score file for {source_exp}. Checked: {', '.join(str(path) for path in candidates)}")


def run_score_ablation(args) -> dict[str, str]:
    from src.gbm.datasets import get_run_dir

    variant_names = [name.strip() for name in str(getattr(args, "variants", "") or "").split(",") if name.strip()]
    if not variant_names:
        variant_names = ["recon_only", "association_kl_only", "recon_plus_kl", "full_default"]
    unknown = sorted(set(variant_names) - set(SCORE_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")

    source_df = load_source_scores(args.source_exp_name)
    selected = {name: SCORE_VARIANTS[name] for name in variant_names}
    enriched = apply_score_variants(source_df, selected)
    run_dir = get_run_dir(args.output_exp_name)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for variant_name in variant_names:
        variant_df = enriched.copy()
        variant_df["score"] = variant_df[f"score_{variant_name}"]
        variant_df["score_variant"] = variant_name
        variant_df["score_variant_label"] = variant_label(variant_name)
        variant_df["source_weights"] = json.dumps(selected[variant_name])
        out_path = reports_dir / f"{variant_name}_test_scores.csv"
        variant_df.to_csv(out_path, index=False)
        paths[variant_name] = str(out_path)

    combined_path = reports_dir / "all_variants_test_scores.csv"
    enriched.to_csv(combined_path, index=False)
    paths["combined"] = str(combined_path)
    save_json(
        run_dir / "score_ablation_manifest.json",
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
    print(f"Saved {len(variant_names)} variant score files under {reports_dir}")
    return paths

__all__ = [
    "BASELINE_NAMES",
    "STATISTICAL_BASELINES",
    "average_precision_score_local",
    "binary_metrics",
    "cusum_abs_return",
    "ewma_volatility",
    "last_return_zscore",
    "load_source_scores",
    "max_return_zscore",
    "mean_abs_return",
    "parse_baselines",
    "rolling_volatility",
    "roc_auc_score_local",
    "run_score_ablation",
    "run_statistical_baselines",
    "score_manifest",
    "summarize_baseline",
    "vol_ratio_short_long",
]
