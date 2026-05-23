#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "gbm" / "run_joint.py"),
        "--exp-name",
        "canonical_gbm_attention",
        *sys.argv[1:],
        "--association-mode",
        "canonical_gbm",
    ]
    completed = subprocess.run(command, cwd=root)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
