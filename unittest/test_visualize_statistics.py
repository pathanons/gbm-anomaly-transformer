from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.gbm.visualize import add_final_score_columns, final_spike_mask, plot_mad_ticker


def test_final_score_columns_and_mad_threshold_are_configurable() -> None:
    frame = pd.DataFrame(
        {
            "nll": [-2.0, -2.1, -2.2, 3.0, -2.0],
            "association_discrepancy": [0.1, 0.1, 0.2, 0.3, 0.1],
        }
    )

    enriched = add_final_score_columns(frame)
    loose_mask, loose_stats = final_spike_mask(enriched, k=1.0)
    strict_mask, strict_stats = final_spike_mask(enriched, k=9.0)

    assert {"raw_sum", "score_robust_z", "delta_score_robust_z"}.issubset(enriched.columns)
    assert loose_stats.threshold < strict_stats.threshold
    assert int(loose_mask.sum()) >= int(strict_mask.sum())


def test_plot_mad_ticker_writes_png_with_custom_k(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "end_date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "nll": [-2.0, -2.1, -2.0, 4.0, -2.2, -2.1],
            "association_discrepancy": [0.1, 0.1, 0.1, 0.4, 0.1, 0.1],
            "true_jump": [0, 0, 0, 1, 0, 0],
        }
    )

    result = plot_mad_ticker(
        scores,
        "AAA",
        tmp_path,
        label_columns=["true_jump"],
        price_dir=tmp_path,
        price_z_thr=3.0,
        mad_k=10.0,
    )

    assert result is not None
    assert Path(result.outpath).exists()
    assert Path(result.outpath).suffix == ".png"
