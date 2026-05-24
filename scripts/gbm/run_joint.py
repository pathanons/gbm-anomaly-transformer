#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.gbm.data import discover_tickers
from src.gbm.paths import get_run_dir


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_handle=None) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    if log_handle is not None:
        log_handle.write(line + "\n")
        log_handle.flush()


def run_command(command, log_handle=None):
    log(f"START {' '.join(command)}", log_handle)
    start = dt.datetime.now()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if log_handle is not None:
            log_handle.write(line)
            log_handle.flush()
    return_code = process.wait()
    elapsed = (dt.datetime.now() - start).total_seconds()
    if return_code != 0:
        log(f"FAIL exit_code={return_code} elapsed_sec={elapsed:.1f}", log_handle)
        raise SystemExit(return_code)
    log(f"DONE exit_code=0 elapsed_sec={elapsed:.1f}", log_handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the joint EXP3 pipeline end to end")
    parser.add_argument("--exp-name", default="experiment3_joint")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
    parser.add_argument("--split-method", default="chronological", choices=["chronological", "random"])
    parser.add_argument("--purge-gap", type=int, default=None, help="Embargo gap in windows between chronological splits")
    parser.add_argument("--include-anomalous-train", action="store_true", help="Allow labeled anomalous windows in training")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--predictive-distribution", default="gaussian", choices=["gaussian", "student_t"])
    parser.add_argument(
        "--association-mode",
        default="gaussian_log_return",
        choices=["gaussian_log_return", "canonical_gbm", "temporal", "none"],
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dist-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--divergence-weight", type=float, default=0.25)
    parser.add_argument("--association-weight", type=float, default=0.1)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--threshold-method",
        default="quantile",
        choices=["quantile", "conformal", "per_ticker_conformal", "evt", "tail_probability", "var"],
    )
    parser.add_argument("--evt-tail-quantile", type=float, default=0.90)
    parser.add_argument("--tolerance-windows", type=int, default=3)
    parser.add_argument("--visualize", action="store_true", help="Generate joint price/score charts after testing")
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--tail-days", type=int, default=None, help="Plot only the latest N calendar days in the scored range")
    parser.add_argument("--spike-percentile", type=float, default=0.95)
    parser.add_argument(
        "--score-mav-windows",
        type=int,
        nargs="+",
        default=[20, 50],
        help="Moving-average window sizes applied to anomaly scores in visualization",
    )
    parser.add_argument("--log-file", default=None, help="Optional path to write a full live pipeline log")
    parser.add_argument("--output-root", default=None, help="Root directory for generated outputs; overrides AT_OUTPUT_ROOT")
    args = parser.parse_args()

    if args.output_root:
        os.environ["AT_OUTPUT_ROOT"] = args.output_root

    tickers = args.tickers if args.tickers else discover_tickers(args.data_path)
    if not tickers:
        raise SystemExit(f"No tickers found in {args.data_path}")

    run_dir = get_run_dir(args.exp_name, args.output_root)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else log_dir / f"joint_pipeline_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    train_common = [
        "--exp-name",
        args.exp_name,
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
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
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
        "--patience",
        str(args.patience),
        "--dist-weight",
        str(args.dist_weight),
        "--recon-weight",
        str(args.recon_weight),
        "--divergence-weight",
        str(args.divergence_weight),
        "--association-weight",
        str(args.association_weight),
    ]
    eval_common = [
        "--exp-name",
        args.exp_name,
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
        train_common.extend(["--purge-gap", str(args.purge_gap)])
        eval_common.extend(["--purge-gap", str(args.purge_gap)])
    if args.include_anomalous_train:
        train_common.append("--include-anomalous-train")
        eval_common.append("--include-anomalous-train")
    if args.normalize_batch:
        train_common.append("--normalize-batch")
        eval_common.append("--normalize-batch")

    with open(log_path, "a", encoding="utf-8") as log_handle:
        log(f"joint experiment={args.exp_name}", log_handle)
        log(f"tickers={len(tickers)} | data_path={args.data_path} | window_size={args.window_size} | batch_size={args.batch_size}", log_handle)
        phases = "train -> validate -> test"
        if args.visualize:
            phases += " -> visualize"
        log(f"phases={phases}", log_handle)
        log(f"selected_tickers={', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}", log_handle)
        log(f"log_file={log_path}", log_handle)

        log("starting train stage", log_handle)
        run_command([sys.executable, "-u", "scripts/gbm/train_joint.py", *train_common, "--tickers", *tickers], log_handle)
        log("starting validation stage", log_handle)
        run_command(
            [
                sys.executable,
                "-u",
                "scripts/gbm/validate_joint.py",
                *eval_common,
                "--threshold-quantile",
                str(args.threshold_quantile),
                "--threshold-method",
                args.threshold_method,
                "--evt-tail-quantile",
                str(args.evt_tail_quantile),
                "--tickers",
                *tickers,
            ],
            log_handle,
        )
        log("starting test stage", log_handle)
        run_command(
            [
                sys.executable,
                "-u",
                "scripts/gbm/test_joint.py",
                *eval_common,
                "--tolerance-windows",
                str(args.tolerance_windows),
                "--tickers",
                *tickers,
            ],
            log_handle,
        )
        if args.visualize:
            log("starting visualization stage", log_handle)
            visualize_cmd = [
                sys.executable,
                "-u",
                "scripts/gbm/visualize_joint.py",
                "--exp-name",
                args.exp_name,
                "--data-path",
                args.data_path,
                "--top-k",
                str(args.top_k),
                "--spike-percentile",
                str(args.spike_percentile),
                "--score-mav-windows",
                *[str(window) for window in args.score_mav_windows],
            ]
            if args.show_true_labels:
                visualize_cmd.append("--show-true-labels")
            if args.full_context:
                visualize_cmd.append("--full-context")
            if args.tail_days is not None:
                visualize_cmd.extend(["--tail-days", str(args.tail_days)])
            run_command(visualize_cmd, log_handle)
        log("pipeline complete", log_handle)

    print(f"[joint] full log saved to {log_path}")


if __name__ == "__main__":
    main()
