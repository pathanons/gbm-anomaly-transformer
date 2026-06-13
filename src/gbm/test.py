"""Canonical testing and score-export surface for GBM experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def _default_attention_score_csv(run_dir: Path, split: str) -> Path:
    if split == "val":
        return run_dir / "reports" / "validation_scores.csv"
    if split == "test_preview":
        return run_dir / "reports" / "test_scores_preview.csv"
    return run_dir / "reports" / "test_scores.csv"


def _parse_optional_ints(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def select_attention_windows(args, run_dir: Path) -> pd.DataFrame:
    split = str(getattr(args, "attention_split", "test") or "test")
    score_csv_value = getattr(args, "attention_score_csv", None)
    score_csv = Path(score_csv_value) if score_csv_value else _default_attention_score_csv(run_dir, split)
    if not score_csv.exists():
        raise FileNotFoundError(f"Missing score CSV for attention selection: {score_csv}")

    frame = pd.read_csv(score_csv)
    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == split].copy()
    if getattr(args, "tickers", None):
        frame = frame[frame["ticker"].astype(str).isin([str(ticker) for ticker in args.tickers])].copy()

    manual_window_ids = _parse_optional_ints(getattr(args, "attention_window_ids", None))
    if manual_window_ids:
        selected = frame[frame["window_id"].astype(int).isin(manual_window_ids)].copy()
    else:
        select_by = str(getattr(args, "attention_select_by", "association_discrepancy") or "association_discrepancy")
        if select_by == "labeled":
            if "y_true" not in frame.columns:
                raise ValueError("attention_select_by=labeled requires y_true in the score CSV")
            selected = frame[frame["y_true"].astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        else:
            sort_column = select_by
            if sort_column not in frame.columns:
                raise ValueError(f"attention_select_by column not found in score CSV: {sort_column}")
            selected = frame.copy()
        top_k = max(int(getattr(args, "attention_top_k", 5) or 5), 1)
        selected = selected.sort_values(sort_column, ascending=False).head(top_k)

    if selected.empty:
        raise ValueError("No attention windows selected")
    return selected.reset_index(drop=True)


def _entropy(values: np.ndarray) -> float:
    safe = np.clip(values.astype(float), 1e-12, None)
    return float(-(safe * np.log(safe)).sum())


def _attention_summary(series: np.ndarray, prior: np.ndarray) -> dict[str, float | int]:
    endpoint_series = series[:, :, -1, :]
    endpoint_prior = prior[:, :, -1, :]
    endpoint_abs_diff = np.abs(endpoint_series - endpoint_prior)
    averaged_endpoint = endpoint_series.mean(axis=(0, 1))
    top_index = int(np.argmax(averaged_endpoint))
    length = int(averaged_endpoint.shape[0])
    return {
        "endpoint_l1_mean": float(endpoint_abs_diff.sum(axis=-1).mean()),
        "endpoint_series_entropy_mean": float(np.mean([_entropy(row) for row in endpoint_series.reshape(-1, length)])),
        "endpoint_prior_entropy_mean": float(np.mean([_entropy(row) for row in endpoint_prior.reshape(-1, length)])),
        "endpoint_top_index": top_index,
        "endpoint_top_lag": int(length - 1 - top_index),
        "endpoint_top_weight": float(averaged_endpoint[top_index]),
    }


def export_attention_artifacts(args) -> pd.DataFrame:
    set_seed(args.seed)
    device = resolve_device(args.device)
    split = str(getattr(args, "attention_split", "test") or "test")
    if split not in {"val", "test"}:
        raise ValueError("attention_split must be val or test")
    print(f"[attention] device={device} split={split}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    selected = select_attention_windows(args, run_dir)
    target_ids = {int(value) for value in selected["window_id"].tolist()}
    print(f"[attention] selected windows={len(target_ids)} from score CSV", flush=True)

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    loader = val_loader if split == "val" else test_loader
    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))
    model.eval()

    out_root_value = getattr(args, "attention_out", None)
    out_root = Path(out_root_value) if out_root_value else run_dir / "reports" / "attention"
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    score_lookup = selected.set_index(selected["window_id"].astype(int)).to_dict(orient="index")
    rows = []
    with torch.no_grad():
        for batch in loader:
            meta = batch["meta"]
            batch_window_ids = [int(value) for value in meta["window_id"]]
            wanted_positions = [idx for idx, window_id in enumerate(batch_window_ids) if window_id in target_ids]
            if not wanted_positions:
                continue

            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            _, _, _, _, attn_maps, _, _, _, association = model(
                x,
                returns=returns,
                time_deltas=time_deltas,
                return_attention=True,
            )
            series_stack = torch.stack([attn["series"] for attn in attn_maps], dim=1).detach().cpu().numpy()
            prior_stack = torch.stack([attn["prior"] for attn in attn_maps], dim=1).detach().cpu().numpy()
            assoc_values = association.detach().cpu().numpy() if association is not None else np.zeros(len(batch_window_ids))

            for idx in wanted_positions:
                window_id = int(batch_window_ids[idx])
                ticker = str(meta["ticker"][idx])
                start_idx = int(meta["start_idx"][idx])
                dates = window_store[ticker]["dates"][start_idx: start_idx + int(args.window_size)]
                series = series_stack[idx].astype(np.float32)
                prior = prior_stack[idx].astype(np.float32)
                diff = (series - prior).astype(np.float32)
                out_path = data_dir / f"{ticker}_window{window_id}.npz"
                np.savez_compressed(
                    out_path,
                    series=series,
                    prior=prior,
                    diff=diff,
                    dates=np.asarray(dates, dtype=str),
                )
                score_row = score_lookup.get(window_id, {})
                row = {
                    "ticker": ticker,
                    "split": split,
                    "window_id": window_id,
                    "start_idx": start_idx,
                    "end_idx": int(meta["end_idx"][idx]),
                    "start_date": meta["start_date"][idx],
                    "end_date": meta["end_date"][idx],
                    "y_true": int(batch["y"][idx].item()),
                    "association_discrepancy": float(assoc_values[idx]),
                    "score": float(score_row.get("score", np.nan)),
                    "nll": float(score_row.get("nll", np.nan)),
                    "association_mode": args.association_mode,
                    "predictive_distribution": args.predictive_distribution,
                    "attention_npz": str(out_path),
                }
                row.update(_attention_summary(series, prior))
                rows.append(row)
                target_ids.remove(window_id)

            if not target_ids:
                break

    if target_ids:
        print(f"[attention] warning: {len(target_ids)} selected windows were not found in {split} loader", flush=True)
    manifest_df = pd.DataFrame(rows)
    manifest_path = out_root / "attention_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[attention] saved manifest={manifest_path}", flush=True)

    if getattr(args, "attention_make_plots", True):
        from src.gbm.visualize import run_attention_visualize

        plot_args = type(
            "AttentionPlotArgs",
            (),
            {
                "attention_manifest": str(manifest_path),
                "out": str(out_root / "figures"),
                "attention_layer": getattr(args, "attention_layer", 0),
                "attention_head": getattr(args, "attention_head", 0),
            },
        )()
        run_attention_visualize(plot_args)
    return manifest_df


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
    "export_attention_artifacts",
    "load_state_dict",
    "score_kwargs",
    "score_summary",
    "select_attention_windows",
    "summarize_events",
    "test_model",
    "validate_model",
]
