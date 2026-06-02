from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import pandas as pd


EVENT_COLORS = {
    "jump": "#2ca02c",
    "drop": "#9467bd",
    "volume_spike": "#17becf",
    "volatility_shock": "#8c564b",
    "regime_shift": "#e377c2",
    "is_anomaly": "#2ca02c",
}


def roc_auc_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    negatives = int((data["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = data["score"].rank(method="average")
    positive_rank_sum = float(ranks[data["y"] == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision_score_local(y_true: pd.Series, y_score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    positives = int((data["y"] == 1).sum())
    if positives == 0:
        return float("nan")
    data = data.sort_values("score", ascending=False).reset_index(drop=True)
    true_positive_count = 0
    precision_sum = 0.0
    for idx, label in enumerate(data["y"], start=1):
        if int(label) == 1:
            true_positive_count += 1
            precision_sum += true_positive_count / idx
    return precision_sum / positives

def event_columns_from_label_file(label_df: pd.DataFrame) -> list[str]:
    event_columns = [column for column in label_df.columns if column not in {"Date", "is_anomaly"}]
    if not event_columns and "is_anomaly" in label_df.columns:
        event_columns = ["is_anomaly"]
    return event_columns


def add_test_event_columns(results: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    frames = []
    all_event_columns = set()
    for ticker, frame in results.groupby("ticker", sort=False):
        frame = frame.copy()
        label_path = data_path / f"{ticker}_anomaly_label.csv"
        if not label_path.exists():
            frames.append(frame)
            continue

        label_df = pd.read_csv(label_path, parse_dates=["Date"])
        event_columns = event_columns_from_label_file(label_df)
        all_event_columns.update(event_columns)
        for event_column in event_columns:
            event_dates = label_df.loc[label_df[event_column].fillna(0).astype(float) > 0, "Date"]
            event_dates = sorted(pd.to_datetime(event_dates.drop_duplicates()))
            values = []
            for _, row in frame.iterrows():
                start_date = pd.to_datetime(row["start_date"])
                end_date = pd.to_datetime(row["end_date"])
                event_idx = bisect_left(event_dates, start_date)
                values.append(int(event_idx < len(event_dates) and event_dates[event_idx] <= end_date))
            frame[f"true_{event_column}"] = values
        frames.append(frame)

    annotated = pd.concat(frames, ignore_index=True)
    for event_column in sorted(all_event_columns):
        output_column = f"true_{event_column}"
        if output_column not in annotated.columns:
            annotated[output_column] = 0
        annotated[output_column] = annotated[output_column].fillna(0).astype(int)
    return annotated
