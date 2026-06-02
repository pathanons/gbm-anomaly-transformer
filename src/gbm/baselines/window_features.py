from __future__ import annotations

import numpy as np


def extract_window_features(x_window: np.ndarray, returns_window: np.ndarray) -> np.ndarray:
    """Hand-crafted features for tree / sklearn baselines."""
    x_window = np.nan_to_num(x_window.astype(float))
    returns_window = np.nan_to_num(returns_window.astype(float))

    abs_ret = np.abs(returns_window)
    feats = [
        float(np.mean(returns_window)),
        float(np.std(returns_window, ddof=0)),
        float(np.min(returns_window)),
        float(np.max(returns_window)),
        float(np.mean(abs_ret)),
        float(np.max(abs_ret)),
        float(np.percentile(abs_ret, 95)),
        float(np.percentile(abs_ret, 99)),
    ]

    if len(returns_window) >= 5:
        short = returns_window[-5:]
        feats.append(float(np.std(short, ddof=0)))
    else:
        feats.append(float(np.std(returns_window, ddof=0)))

    for col in range(x_window.shape[1]):
        channel = x_window[:, col]
        feats.extend(
            [
                float(np.mean(channel)),
                float(np.std(channel, ddof=0)),
                float(np.min(channel)),
                float(np.max(channel)),
            ]
        )

    return np.asarray(feats, dtype=np.float64)
