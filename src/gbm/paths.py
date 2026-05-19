#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def get_output_root(output_root: Optional[str | Path] = None) -> Path:
    root = output_root or os.environ.get("AT_OUTPUT_ROOT") or "results"
    return Path(root).expanduser()


def get_run_dir(exp_name: str, output_root: Optional[str | Path] = None) -> Path:
    return get_output_root(output_root) / "experiments" / exp_name