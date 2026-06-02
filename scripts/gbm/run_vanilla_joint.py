#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
    parser = argparse.ArgumentParser(description="Run the vanilla Anomaly Transformer pipeline end to end")
    parser.add_argument("--exp-name", default="experiment3_vanilla_joint")
    parser.add_argument("--data-path", default="datasets/SP500_event_taxonomy_w100")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--features", default="all", choices=["all", "price_only", "volume_only"])
    parser.add_argument("--normalize-batch", action="store_true")
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--prior-type", default="gaussian", choices=["gaussian", "powerlaw"])
    parser.add_argument("--visualize", action="store_true", help="Generate vanilla price/score charts after testing")
    parser.add_argument("--show-true-labels", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--full-context", action="store_true")
    parser.add_argument("--spike-percentile", type=float, default=0.95)
    parser.add_argument("--log-file", default=None, help="Optional path to write a full live pipeline log")
    parser.add_argument("--output-root", default=None, help="Root directory for generated outputs; overrides AT_OUTPUT_ROOT")
    args = parser.parse_args()

    if args.output_root:
        os.environ["AT_OUTPUT_ROOT"] = args.output_root

    root = Path(__file__).resolve().parents[2]
    tickers = args.tickers if args.tickers else []

    run_dir = get_run_dir(args.exp_name, args.output_root)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else log_dir / f"vanilla_pipeline_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    train_common = [
        "--exp-name",
        args.exp_name,
        "--data-path",
        args.data_path,
        "--device",
        args.device,
        "--features",
        args.features,
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
        "--patience",
        str(args.patience),
        "--k",
        str(args.k),
        "--temperature",
        str(args.temperature),
        "--prior-type",
        args.prior_type,
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
        "--temperature",
        str(args.temperature),
        "--prior-type",
        args.prior_type,
    ]
    if args.normalize_batch:
        train_common.append("--normalize-batch")
        eval_common.append("--normalize-batch")

    with open(log_path, "a", encoding="utf-8") as log_handle:
        log(f"vanilla experiment={args.exp_name}", log_handle)
        log(f"data_path={args.data_path} | window_size={args.window_size} | batch_size={args.batch_size} | prior_type={args.prior_type}", log_handle)
        log(f"selected_tickers={'all' if not tickers else ', '.join(tickers[:10])}{'...' if tickers and len(tickers) > 10 else ''}", log_handle)
        phases = "train -> validate -> test"
        if args.visualize:
            phases += " -> visualize"
        log(f"phases={phases}", log_handle)
        log(f"log_file={log_path}", log_handle)

        train_cmd = [sys.executable, "-u", str(root / "scripts" / "gbm" / "train_vanilla_joint.py"), *train_common]
        if tickers:
            train_cmd.extend(["--tickers", *tickers])
        run_command(train_cmd, log_handle)

        validate_cmd = [sys.executable, "-u", str(root / "scripts" / "gbm" / "validate_vanilla_joint.py"), *eval_common]
        if tickers:
            validate_cmd.extend(["--tickers", *tickers])
        run_command(validate_cmd, log_handle)

        test_cmd = [sys.executable, "-u", str(root / "scripts" / "gbm" / "test_vanilla_joint.py"), *eval_common]
        if tickers:
            test_cmd.extend(["--tickers", *tickers])
        run_command(test_cmd, log_handle)

        if args.visualize:
            visualize_cmd = [
                sys.executable,
                "-u",
                str(root / "scripts" / "gbm" / "visualize_vanilla_joint.py"),
                "--exp-name",
                args.exp_name,
                "--data-path",
                args.data_path,
                "--top-k",
                str(args.top_k),
                "--spike-percentile",
                str(args.spike_percentile),
            ]
            if args.show_true_labels:
                visualize_cmd.append("--show-true-labels")
            if args.full_context:
                visualize_cmd.append("--full-context")
            run_command(visualize_cmd, log_handle)

        log("pipeline complete", log_handle)

    print(f"[vanilla] full log saved to {log_path}")


if __name__ == "__main__":
    main()
