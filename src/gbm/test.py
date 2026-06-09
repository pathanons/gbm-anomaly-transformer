"""Canonical testing and score-export surface for GBM experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.gbm.datasets import get_run_dir, save_json, set_seed
from src.gbm.score import add_test_event_columns, average_precision_score_local, collect_joint_scores, roc_auc_score_local
from src.gbm.train import build_gbm_model, build_runtime_loaders, checkpoint_path, load_state_dict, resolve_device, score_kwargs


def score_summary(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
    score = frame["score"].astype(float)
    y_true = frame["y_true"].astype(int)
    output: dict[str, float | int] = {
        f"{prefix}n_windows": int(len(frame)),
        f"{prefix}anomaly_rate": float(y_true.mean()) if len(frame) else 0.0,
        f"{prefix}score_mean": float(score.mean()) if len(frame) else float("nan"),
        f"{prefix}score_std": float(score.std(ddof=0)) if len(frame) else float("nan"),
        f"{prefix}score_min": float(score.min()) if len(frame) else float("nan"),
        f"{prefix}score_max": float(score.max()) if len(frame) else float("nan"),
        f"{prefix}score_p50": float(score.quantile(0.50)) if len(frame) else float("nan"),
        f"{prefix}score_p90": float(score.quantile(0.90)) if len(frame) else float("nan"),
        f"{prefix}score_p95": float(score.quantile(0.95)) if len(frame) else float("nan"),
        f"{prefix}score_p99": float(score.quantile(0.99)) if len(frame) else float("nan"),
    }
    if y_true.nunique() >= 2:
        output[f"{prefix}roc_auc_from_score"] = float(roc_auc_score_local(y_true, score))
        output[f"{prefix}pr_auc_from_score"] = float(average_precision_score_local(y_true, score))
    else:
        output[f"{prefix}roc_auc_from_score"] = float("nan")
        output[f"{prefix}pr_auc_from_score"] = float("nan")
    return output


def summarize_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_columns = [column for column in frame.columns if column.startswith("true_")]
    for event_column in event_columns:
        event_mask = frame[event_column].fillna(0).astype(int) > 0
        normal_mask = ~event_mask
        row = {
            "event_type": event_column.replace("true_", ""),
            "event_windows": int(event_mask.sum()),
            "non_event_windows": int(normal_mask.sum()),
        }
        if event_mask.any():
            row.update(
                {
                    "event_score_mean": float(frame.loc[event_mask, "score"].mean()),
                    "event_score_p95": float(frame.loc[event_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"event_score_mean": float("nan"), "event_score_p95": float("nan")})
        if normal_mask.any():
            row.update(
                {
                    "non_event_score_mean": float(frame.loc[normal_mask, "score"].mean()),
                    "non_event_score_p95": float(frame.loc[normal_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"non_event_score_mean": float("nan"), "non_event_score_p95": float("nan")})
        rows.append(row)
    return pd.DataFrame(rows)


def validate_model(args) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[validate] device={device}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[validate] loading checkpoint={model_path}")
    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[validate] windows total={len(manifest)} | val={len(val_ds)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    val_df = collect_joint_scores(model, val_loader, device, phase="validate", **score_kwargs(args))
    test_df = collect_joint_scores(model, test_loader, device, phase="test_preview", **score_kwargs(args))
    if val_df.empty:
        raise RuntimeError("Validation scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dist_weight": args.dist_weight,
        "recon_weight": args.recon_weight,
        "divergence_weight": args.divergence_weight,
        "association_weight": args.association_weight,
        "score_mode": args.score_mode,
        "quantile_count": args.quantile_count,
        "tail_weight_gamma": args.tail_weight_gamma,
        "tail_weight_power": args.tail_weight_power,
        "predictive_distribution": args.predictive_distribution,
        "association_mode": args.association_mode,
        "val_score_mean": float(val_df["score"].mean()),
        "val_score_std": float(val_df["score"].std(ddof=0)),
        "test_preview_score_mean": float(test_df["score"].mean()),
        "test_preview_score_std": float(test_df["score"].std(ddof=0)),
        "val_windows": int(len(val_df)),
        "test_preview_windows": int(len(test_df)),
        "ticker_count": int(manifest["ticker"].nunique()),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }
    save_json(reports_dir / "score_summary.json", summary)
    val_df.to_csv(reports_dir / "validation_scores.csv", index=False)
    test_df.to_csv(reports_dir / "test_scores_preview.csv", index=False)
    print(f"Saved validation summary to {reports_dir / 'score_summary.json'}")
    return summary


def test_model(args) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test] device={device}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[test] loading checkpoint={model_path}")
    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[test] windows total={len(manifest)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    test_df = collect_joint_scores(model, test_loader, device, phase="test", **score_kwargs(args))
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "test_scores.csv"

    if getattr(args, "fast_export", False):
        test_df.to_csv(scores_path, index=False)
        print(f"Saved raw test scores to {scores_path}")
        return {"scores": str(scores_path)}

    test_df = add_test_event_columns(test_df, Path(args.data_path))
    metrics = score_summary(test_df)
    metrics.update(
        {
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "score_mode": args.score_mode,
            "ticker_count": int(manifest["ticker"].nunique()),
            "mean_reconstruction_error": float(test_df["reconstruction_error"].mean()),
            "mean_nll": float(test_df["nll"].mean()),
            "mean_divergence": float(test_df["divergence"].mean()),
            "mean_association_discrepancy": float(test_df["association_discrepancy"].mean()),
        }
    )

    metrics_path = reports_dir / "score_metrics.json"
    test_df.to_csv(scores_path, index=False)
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = []
    for ticker, group in test_df.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"{ticker}_test_scores.csv", index=False)
        row = {"ticker": ticker}
        row.update(score_summary(group))
        ticker_rows.append(row)
    pd.DataFrame(ticker_rows).to_csv(reports_dir / "score_metrics_by_ticker.csv", index=False)

    event_summary = summarize_events(test_df)
    if not event_summary.empty:
        event_summary.to_csv(reports_dir / "score_summary_by_event_type.csv", index=False)

    print(f"Saved raw test scores to {scores_path}")
    print(f"Saved raw score metrics to {metrics_path}")
    return metrics


__all__ = [
    "build_gbm_model",
    "build_runtime_loaders",
    "checkpoint_path",
    "collect_joint_scores",
    "load_state_dict",
    "score_kwargs",
    "score_summary",
    "summarize_events",
    "test_model",
    "validate_model",
]
