#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

PYTHON_BIN="${PYTHON:-python3}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-experiment3_joint}"
DATA_PATH="${DATA_PATH:-datasets/SP500_event_taxonomy_w100}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
FEATURES="${FEATURES:-all}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"

exec "$PYTHON_BIN" -u scripts/gbm/run_joint.py \
  --exp-name "$EXP_NAME" \
  --data-path "$DATA_PATH" \
  --window-size "$WINDOW_SIZE" \
  --features "$FEATURES" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  "$@"
