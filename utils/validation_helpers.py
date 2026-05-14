"""Dataset validation helpers.

Use this module for:
- selecting label columns from a dataset
- lightweight alignment and anomaly bookkeeping
- shared validation logic between scripts and reports
"""

from __future__ import annotations

from typing import List

import pandas as pd


LABEL_CANDIDATES = ["jump", "drop", "volume_spike", "volatility_shock", "regime_shift", "criterion_a", "criterion_b", "criterion_c"]


def detect_label_columns(df: pd.DataFrame) -> List[str]:
    preferred = [col for col in LABEL_CANDIDATES if col in df.columns]
    if preferred:
        return preferred
    return [col for col in df.columns if col not in {"Date", "regime", "is_anomaly"} and pd.api.types.is_numeric_dtype(df[col])]
