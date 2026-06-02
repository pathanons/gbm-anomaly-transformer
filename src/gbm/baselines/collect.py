from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def rows_from_scores(
    baseline_name: str,
    meta_batch: dict,
    y_batch: torch.Tensor,
    scores: torch.Tensor,
    extra: dict[str, torch.Tensor] | None = None,
) -> list[dict]:
    rows = []
    extra = extra or {}
    for idx in range(len(scores)):
        row = {
            "baseline": baseline_name,
            "ticker": meta_batch["ticker"][idx],
            "split": meta_batch["split"][idx],
            "window_id": int(meta_batch["window_id"][idx]),
            "start_idx": int(meta_batch["start_idx"][idx]),
            "end_idx": int(meta_batch["end_idx"][idx]),
            "start_date": meta_batch["start_date"][idx],
            "end_date": meta_batch["end_date"][idx],
            "y_true": int(y_batch[idx].item()),
            "score": float(scores[idx].item()),
        }
        for key, tensor in extra.items():
            row[key] = float(tensor[idx].item())
        rows.append(row)
    return rows


def collect_reconstruction_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline_name: str,
    phase: str,
) -> pd.DataFrame:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    rows: list[dict] = []
    batch_total = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            recon = model(x)
            recon_error = criterion(recon, x).mean(dim=(1, 2))

            if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                print(f"    [{phase}] batch {batch_idx}/{batch_total} | rows={len(recon_error)}", flush=True)

            rows.extend(
                rows_from_scores(
                    baseline_name,
                    batch["meta"],
                    batch["y"],
                    recon_error,
                    extra={"reconstruction_error": recon_error},
                )
            )

    return pd.DataFrame(rows)


def collect_loader_batches(loader: DataLoader) -> Iterable[dict]:
    for batch in loader:
        yield batch
