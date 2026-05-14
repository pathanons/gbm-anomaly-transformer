from __future__ import annotations

import torch


def is_mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Resolve cuda/mps/cpu with an Apple Silicon friendly auto mode."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable; falling back to CPU", flush=True)
            return torch.device("cpu")
        return torch.device("cuda")

    if requested == "mps":
        if not is_mps_available():
            print("[device] MPS requested but unavailable; falling back to CPU", flush=True)
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device(requested)
