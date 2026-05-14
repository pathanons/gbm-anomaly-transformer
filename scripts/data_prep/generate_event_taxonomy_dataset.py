#!/usr/bin/env python3
"""
Generate an event-taxonomy version of the SP500 dataset.

This script mirrors the OHLCV files into a new dataset directory and writes
event labels for each ticker using an explicit, windowed financial taxonomy:
jump, drop, volume_spike, volatility_shock, and regime_shift.

The generated output is window-aware. The output directory name should include
the chosen lookback window, for example:
    datasets/SP500_event_taxonomy_w100

The event labels are computed using only historical data up to t-1.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.dataset_io import discover_sp500_tickers, read_ohlcv_csv, write_dataframe_csv
from utils.event_taxonomy import LabelConfig, build_event_labels, render_spec_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a taxonomy-labeled SP500 dataset")
    parser.add_argument("--input-path", default="datasets/SP500", help="Source SP500 dataset directory")
    parser.add_argument("--output-path", default=None, help="Output dataset directory")
    parser.add_argument("--lookback-window", type=int, default=100, help="Main rolling lookback window W")
    parser.add_argument("--return-z", type=float, default=2.5, help="Z threshold for return shocks")
    parser.add_argument("--volume-z", type=float, default=2.5, help="Z threshold for volume spikes")
    parser.add_argument("--volatility-z", type=float, default=2.0, help="Z threshold for volatility shocks")
    parser.add_argument("--regime-mean-z", type=float, default=1.5, help="Threshold for regime mean shift")
    parser.add_argument("--regime-vol-ratio", type=float, default=1.5, help="Threshold for regime volatility ratio")
    parser.add_argument("--regime-persistence", type=int, default=None, help="Persistence in days for regime shifts")
    parser.add_argument("--copy-ohlcv", action="store_true", help="Copy OHLCV files into the output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("datasets") / f"SP500_event_taxonomy_w{args.lookback_window}"

    output_path.mkdir(parents=True, exist_ok=True)
    label_config = LabelConfig(
        lookback_window=args.lookback_window,
        short_window=max(5, int(round(args.lookback_window / 5))),
        regime_persistence=args.regime_persistence or max(3, int(round(args.lookback_window / 20))),
        return_z=args.return_z,
        volume_z=args.volume_z,
        volatility_z=args.volatility_z,
        regime_mean_z=args.regime_mean_z,
        regime_vol_ratio=args.regime_vol_ratio,
    )

    tickers = discover_sp500_tickers(input_path)
    if not tickers:
        raise FileNotFoundError(f"No SP500 OHLCV files found in {input_path}")

    summary_rows: List[Dict[str, object]] = []
    total_rows = 0
    total_events = 0

    for ticker in tickers:
        ohlcv_path = input_path / f"{ticker}_ohlcv.csv"
        if not ohlcv_path.exists():
            continue

        ohlcv = read_ohlcv_csv(ohlcv_path)
        labels, stats = build_event_labels(ohlcv, label_config)

        if args.copy_ohlcv:
            shutil.copy2(ohlcv_path, output_path / ohlcv_path.name)

        label_path = output_path / f"{ticker}_anomaly_label.csv"
        write_dataframe_csv(labels, label_path)

        summary_rows.append({"ticker": ticker, **stats})
        total_rows += int(stats["rows"])
        total_events += int(stats["event_count"])

    manifest = {
        "source_path": str(input_path),
        "output_path": str(output_path),
        "event_taxonomy": ["jump", "drop", "volume_spike", "volatility_shock", "regime_shift"],
        "label_config": asdict(label_config),
        "tickers": len(summary_rows),
        "rows": total_rows,
        "event_count": total_events,
        "composite_anomaly_rate": float(total_events / total_rows) if total_rows else 0.0,
    }

    pd.DataFrame(summary_rows).to_csv(output_path / "event_label_summary.csv", index=False)
    (output_path / "dataset_spec.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_path / "LABEL_SPEC.md").write_text(render_spec_markdown(label_config), encoding="utf-8")

    print(f"Generated taxonomy dataset at {output_path}")
    print(f"Tickers: {len(summary_rows)}")
    print(f"Rows: {total_rows}")
    print(f"Composite anomaly rate: {manifest['composite_anomaly_rate']:.6f}")
if __name__ == "__main__":
    main()
