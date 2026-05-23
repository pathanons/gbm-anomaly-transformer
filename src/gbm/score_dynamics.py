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




def add_score_delta_columns(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker, frame in scores.groupby("ticker", sort=False):
        frame = frame.sort_values("end_date").reset_index(drop=True).copy()
        frame["score_delta"] = frame["score"].diff()
        frame["score_delta_abs"] = frame["score_delta"].abs()
        frame["score_curvature"] = frame["score_delta"].diff()
        frame["score_curvature_abs"] = frame["score_curvature"].abs()
        frame["score_next_delta"] = frame["score"].shift(-1) - frame["score"]
        frame["score_turning_point"] = ((frame["score_delta"] * frame["score_next_delta"]) < 0).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else scores.copy()




def apply_score_turning_point_rule(scores: pd.DataFrame) -> pd.DataFrame:
    result = add_score_delta_columns(scores)
    result["score_change_threshold"] = None
    result["score_change_column"] = "score_turning_point"
    result["score_change_pred"] = result["score_turning_point"].astype(int)
    result["y_pred"] = result["score_change_pred"]
    return result

def apply_score_change_threshold(scores: pd.DataFrame, threshold: float, column: str = "score_curvature_abs") -> pd.DataFrame:
    result = add_score_delta_columns(scores)
    if column not in result.columns:
        raise ValueError(f"Unknown score-change column: {column}")
    result["score_change_threshold"] = float(threshold)
    result["score_change_column"] = column
    result["score_change_pred"] = (result[column].fillna(0.0) > threshold).astype(int)
    result["y_pred"] = result["score_change_pred"]
    return result

def add_score_change_flags(
    scores: pd.DataFrame,
    rolling_window: int,
    min_periods: int,
    mad_k: float,
    fallback_quantile: float,
) -> pd.DataFrame:
    frames = []
    for ticker, frame in scores.groupby("ticker", sort=False):
        frame = frame.sort_values("end_date").reset_index(drop=True).copy()
        frame["score_delta"] = frame["score"].diff()
        frame["score_delta_abs"] = frame["score_delta"].abs()

        rolling = frame["score_delta_abs"].shift(1).rolling(rolling_window, min_periods=min_periods)
        rolling_median = rolling.median()
        rolling_mad = rolling.apply(lambda x: (x - x.median()).abs().median(), raw=False)
        frame["dynamic_threshold"] = rolling_median + mad_k * 1.4826 * rolling_mad

        fallback_threshold = float(frame["score_delta_abs"].dropna().quantile(fallback_quantile))
        frame["dynamic_threshold"] = frame["dynamic_threshold"].fillna(fallback_threshold)
        frame["score_change_pred"] = (frame["score_delta_abs"] > frame["dynamic_threshold"]).astype(int)
        frame["score_change_rank"] = frame["score_delta_abs"].rank(method="first", ascending=False)
        frame["threshold_source"] = f"rolling_median_plus_{mad_k:g}_mad"
        frame.loc[frame["score_delta_abs"].isna(), "score_change_pred"] = 0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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


def metrics_from_dynamic_flags(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].astype(int)
    y_pred = frame["score_change_pred"].astype(int)
    score = frame["score_delta_abs"].fillna(0.0)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0

    if y_true.nunique() < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score_local(y_true, score))
        pr_auc = float(average_precision_score_local(y_true, score))

    return {
        "roc_auc_from_abs_score_change": roc_auc,
        "pr_auc_from_abs_score_change": pr_auc,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tolerance_metrics(frame: pd.DataFrame, tolerance_windows: int) -> dict[str, float]:
    flagged_total = 0
    flagged_correct = 0
    true_total = 0
    true_caught = 0

    for _, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("end_date").reset_index(drop=True)
        true_series = group["y_true"].astype(int)
        flagged_series = group["score_change_pred"].astype(int)
        window = tolerance_windows * 2 + 1

        true_total += int(true_series.sum())
        flagged_total += int(flagged_series.sum())

        nearby_true = true_series.rolling(window=window, center=True, min_periods=1).max().astype(int)
        nearby_flag = flagged_series.rolling(window=window, center=True, min_periods=1).max().astype(int)
        flagged_correct += int(((flagged_series > 0) & (nearby_true > 0)).sum())
        true_caught += int(((true_series > 0) & (nearby_flag > 0)).sum())

    precision = flagged_correct / flagged_total if flagged_total else 0.0
    recall = true_caught / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tolerance_windows": int(tolerance_windows),
        "tolerance_precision": precision,
        "tolerance_recall": recall,
        "tolerance_f1": f1,
        "tolerance_flagged_correct": int(flagged_correct),
        "tolerance_true_caught": int(true_caught),
    }


def summarize_by_event_type(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_columns = [column for column in results.columns if column.startswith("true_") and column != "true_anomaly"]
    for column in sorted(event_columns):
        event_mask = results[column].fillna(0).astype(int) > 0
        flagged = results["score_change_pred"].astype(int) > 0
        total = int(event_mask.sum())
        caught = int((event_mask & flagged).sum())
        missed = int((event_mask & ~flagged).sum())
        rows.append(
            {
                "event_type": column.replace("true_", "", 1),
                "test_windows_with_event": total,
                "caught_by_score_change": caught,
                "missed_by_score_change": missed,
                "catch_rate": caught / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)
