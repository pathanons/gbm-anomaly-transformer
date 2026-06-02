#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.gbm.baselines.collect import collect_reconstruction_scores
from src.gbm.baselines.neural import CNN1DLSTMAutoencoder, LSTMAutoencoder, MLPAutoencoder
from src.gbm.baselines.sklearn_models import (
    build_feature_matrix,
    fit_isolation_forest,
    fit_random_forest,
    score_isolation_forest,
    score_random_forest,
)
from src.gbm.baselines.statistical import STATISTICAL_BASELINES, score_manifest
from src.gbm.baselines.train import train_reconstruction_model
from src.gbm.data import JointWindowDataset, build_joint_loaders, discover_tickers, get_run_dir, save_joint_manifest, set_seed
from src.gbm.device import resolve_device
from src.gbm.io import save_json
from src.gbm.score_dynamics import average_precision_score_local, roc_auc_score_local

NEURAL_BASELINES = {"lstm_autoencoder", "mlp_autoencoder", "cnn1d_lstm_autoencoder"}
SKLEARN_BASELINES = {"random_forest", "isolation_forest"}
ALL_BASELINES = sorted(set(STATISTICAL_BASELINES) | NEURAL_BASELINES | SKLEARN_BASELINES)


def parse_baselines(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if "all" in names:
        return ALL_BASELINES.copy()
    unknown = sorted(set(names) - set(ALL_BASELINES))
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown)}. Available: {', '.join(ALL_BASELINES)}")
    return names


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
    }
    if y_true.nunique() >= 2:
        output[f"{prefix}roc_auc_from_score"] = float(roc_auc_score_local(y_true, score))
        output[f"{prefix}pr_auc_from_score"] = float(average_precision_score_local(y_true, score))
    else:
        output[f"{prefix}roc_auc_from_score"] = float("nan")
        output[f"{prefix}pr_auc_from_score"] = float("nan")
    return output


def save_baseline_outputs(
    run_dir: Path,
    baseline_name: str,
    scores: pd.DataFrame,
    extra_metrics: dict | None = None,
) -> dict[str, str]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_path = reports_dir / f"{baseline_name}_scores.csv"
    scores.to_csv(all_path, index=False)

    test_scores = scores.loc[scores["split"] == "test"].copy()
    test_path = reports_dir / f"{baseline_name}_test_scores.csv"
    test_scores.to_csv(test_path, index=False)

    metrics = score_summary(test_scores)
    metrics["baseline"] = baseline_name
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_path = reports_dir / f"{baseline_name}_metrics.json"
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in test_scores.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"{baseline_name}_{ticker}_test_scores.csv", index=False)

    return {
        "scores": str(all_path),
        "test_scores": str(test_path),
        "metrics": str(metrics_path),
    }


def run_statistical(
    baseline_name: str,
    manifest: pd.DataFrame,
    window_store: dict,
    run_dir: Path,
) -> dict[str, str]:
    print(f"[baselines] statistical scoring: {baseline_name}", flush=True)
    scores = score_manifest(manifest, window_store, baseline_name, STATISTICAL_BASELINES[baseline_name])
    return save_baseline_outputs(run_dir, baseline_name, scores)


def run_neural(
    baseline_name: str,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    window_store: dict,
    scaler,
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    device: torch.device,
) -> dict[str, str]:
    print(f"[baselines] training neural baseline: {baseline_name}", flush=True)
    if baseline_name == "lstm_autoencoder":
        model = LSTMAutoencoder(
            n_features=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif baseline_name == "mlp_autoencoder":
        model = MLPAutoencoder(
            window_size=args.window_size,
            n_features=input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
    elif baseline_name == "cnn1d_lstm_autoencoder":
        model = CNN1DLSTMAutoencoder(
            n_features=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            kernel_size=args.cnn_kernel_size,
        ).to(device)
    else:
        raise ValueError(baseline_name)

    checkpoint_path = run_dir / "models" / f"{baseline_name}.pt"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    train_info = train_reconstruction_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    save_json(run_dir / "logs" / f"{baseline_name}_training.json", train_info)

    all_rows = []
    for split, loader in [
        ("train", train_loader),
        ("val", val_loader),
    ]:
        split_scores = collect_reconstruction_scores(model, loader, device, baseline_name, split)
        all_rows.append(split_scores)

    test_ds = JointWindowDataset(
        manifest,
        window_store,
        scaler,
        args.window_size,
        args.normalize_batch,
        "test",
        train_normal_only=False,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    all_rows.append(collect_reconstruction_scores(model, test_loader, device, baseline_name, "test"))
    scores = pd.concat(all_rows, ignore_index=True)
    return save_baseline_outputs(
        run_dir,
        baseline_name,
        scores,
        extra_metrics={"best_val_loss": train_info["best_val_loss"], "checkpoint": str(checkpoint_path)},
    )


def run_sklearn_baseline(
    baseline_name: str,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    window_store: dict,
    scaler,
    run_dir: Path,
) -> dict[str, str]:
    print(f"[baselines] sklearn baseline: {baseline_name}", flush=True)
    if baseline_name == "random_forest":
        train_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "train",
            train_normal_only=False,
        )
        val_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "val",
            train_normal_only=False,
        )
        test_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "test",
            train_normal_only=False,
        )
        train_meta_x, train_y = build_feature_matrix(train_ds)
        model = fit_random_forest(train_meta_x, train_y, n_estimators=args.rf_estimators, random_state=args.seed)
        score_fn = score_random_forest
        fit_note = "supervised RF on train split (includes labeled anomalies)"
    else:
        train_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "train",
            train_normal_only=True,
        )
        val_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "val",
            train_normal_only=False,
        )
        test_ds = JointWindowDataset(
            manifest,
            window_store,
            scaler,
            args.window_size,
            args.normalize_batch,
            "test",
            train_normal_only=False,
        )
        train_meta_x, _ = build_feature_matrix(train_ds)
        model = fit_isolation_forest(train_meta_x, contamination=args.if_contamination, random_state=args.seed)
        score_fn = score_isolation_forest
        fit_note = "unsupervised IsolationForest on normal train windows only"

    rows = []
    for split, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        meta_x, _ = build_feature_matrix(dataset)
        split_scores = score_fn(model, meta_x)
        for idx, score in enumerate(split_scores):
            row = meta_x.iloc[idx]
            rows.append(
                {
                    "baseline": baseline_name,
                    "ticker": row["ticker"],
                    "split": split,
                    "window_id": int(row["window_id"]),
                    "start_idx": int(row["start_idx"]),
                    "end_idx": int(row["end_idx"]),
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "y_true": int(row["y_true"]),
                    "score": float(score),
                }
            )

    scores = pd.DataFrame(rows)
    return save_baseline_outputs(run_dir, baseline_name, scores, extra_metrics={"fit_protocol": fit_note})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBM-window baseline models (statistical, neural, sklearn)")
    parser.add_argument("--exp-name", default="baseline_comparison")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-kernel-size", type=int, default=3)
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--if-contamination", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--baselines",
        default="all",
        help="Comma-separated baseline names or 'all'. "
        f"Available: {', '.join(ALL_BASELINES)}",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip baselines whose test score CSV already exists")
    args = parser.parse_args()

    set_seed(args.seed)
    selected = parse_baselines(args.baselines)
    tickers = args.tickers if args.tickers else discover_tickers(args.data_path)
    if not tickers:
        raise SystemExit(f"No tickers found in {args.data_path}")

    run_dir = get_run_dir(args.exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[baselines] device={device} | exp={args.exp_name}", flush=True)

    needs_neural = any(name in NEURAL_BASELINES for name in selected)
    needs_loaders = needs_neural or any(name in SKLEARN_BASELINES for name in selected)

    manifest, window_store, scaler = None, None, None
    train_loader = val_loader = None
    input_dim = None

    if needs_loaders:
        manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_joint_loaders(
            data_path=args.data_path,
            tickers=tickers,
            window_size=args.window_size,
            batch_size=args.batch_size,
            step=args.step,
            features=args.features,
            normalize_batch=args.normalize_batch,
            seed=args.seed,
            train_normal_only=True,
        )
        manifest_path = save_joint_manifest(
            run_dir,
            manifest,
            args.seed,
            args.window_size,
            args.step,
            args.features,
        )
        print(f"[baselines] manifest={manifest_path} | windows={len(manifest)}", flush=True)
    else:
        from src.gbm.data import create_joint_manifest

        manifest, window_store, scaler = create_joint_manifest(
            data_path=args.data_path,
            tickers=tickers,
            window_size=args.window_size,
            step=args.step,
            features=args.features,
            seed=args.seed,
        )

    report_paths: dict[str, dict[str, str]] = {}
    summary_rows = []

    for baseline_name in selected:
        test_path = run_dir / "reports" / f"{baseline_name}_test_scores.csv"
        if args.skip_existing and test_path.exists():
            print(f"[baselines] skip existing: {baseline_name}", flush=True)
            report_paths[baseline_name] = {"test_scores": str(test_path)}
            continue

        if baseline_name in STATISTICAL_BASELINES:
            paths = run_statistical(baseline_name, manifest, window_store, run_dir)
        elif baseline_name in NEURAL_BASELINES:
            paths = run_neural(
                baseline_name,
                args,
                manifest,
                window_store,
                scaler,
                input_dim,
                train_loader,
                val_loader,
                run_dir,
                device,
            )
        elif baseline_name in SKLEARN_BASELINES:
            paths = run_sklearn_baseline(baseline_name, args, manifest, window_store, scaler, run_dir)
        else:
            raise SystemExit(f"Unhandled baseline: {baseline_name}")

        report_paths[baseline_name] = paths
        metrics = json.loads(Path(paths["metrics"]).read_text(encoding="utf-8"))
        summary_rows.append(metrics)

    summary_path = run_dir / "reports" / "baselines_summary.csv"
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False))

    manifest_out = run_dir / "baselines_manifest.json"
    save_json(
        manifest_out,
        {
            "exp_name": args.exp_name,
            "data_path": args.data_path,
            "tickers": list(tickers),
            "seed": args.seed,
            "window_size": args.window_size,
            "step": args.step,
            "features": args.features,
            "baselines": selected,
            "reports": report_paths,
            "summary": str(summary_path) if summary_rows else None,
            "protocol": {
                "statistical": "return-based window scores, no training",
                "neural": "unsupervised reconstruction error; train on normal windows only",
                "random_forest": "supervised window features; train split includes labeled anomalies",
                "isolation_forest": "unsupervised window features; fit on normal train windows only",
            },
        },
    )
    print(f"Saved manifest to {manifest_out}")


if __name__ == "__main__":
    main()
