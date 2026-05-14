#!/usr/bin/env python3
"""Generate all SP500 event-taxonomy datasets for the standard window sizes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


WINDOW_SIZES = [50, 100, 150, 200, 250]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    for window_size in WINDOW_SIZES:
        output_path = repo_root / "datasets" / f"SP500_event_taxonomy_w{window_size}"
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "data_prep" / "generate_event_taxonomy_dataset.py"),
            "--input-path",
            str(repo_root / "datasets" / "SP500"),
            "--output-path",
            str(output_path),
            "--lookback-window",
            str(window_size),
            "--copy-ohlcv",
        ]
        print(f"Generating W={window_size} -> {output_path}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"Generation failed for W={window_size}")


if __name__ == "__main__":
    main()
