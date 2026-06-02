from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_reconstruction_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    checkpoint_path,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    wait = 0
    history = []

    def run_epoch(loader: DataLoader, train: bool) -> float:
        model.train(train)
        total = 0.0
        count = 0
        phase = "train" if train else "val"
        batch_total = len(loader)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch_idx, batch in enumerate(loader, start=1):
                x = batch["x"].to(device)
                recon = model(x)
                loss = criterion(recon, x)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += float(loss.item())
                count += 1
                if batch_idx == 1 or batch_idx == batch_total or batch_idx % max(1, batch_total // 5) == 0:
                    print(f"    [{phase}] batch {batch_idx}/{batch_total} | loss={float(loss.item()):.6f}", flush=True)
        return total / max(1, count)

    for epoch in range(epochs):
        print(f"[baseline_train] epoch {epoch + 1}/{epochs}", flush=True)
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return {"best_val_loss": best_val, "history": history}
