"""Canonical testing and score-export surface for GBM experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.gbm.datasets import get_feature_columns, get_run_dir, save_json, set_seed
from src.gbm.score import add_test_event_columns, average_precision_score_local, binary_metrics, collect_joint_scores, roc_auc_score_local
from src.gbm.train import build_gbm_model, build_runtime_loaders, checkpoint_path, load_state_dict, resolve_device, score_kwargs


def score_summary(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
    score = frame["score"].astype(float)
    y_true = frame["y_true"].astype(int)
    output: dict[str, float | int] = {
        f"{prefix}n_windows": int(len(frame)),
        f"{prefix}anomaly_rate": float(y_true.mean()) if len(frame) else 0.0,
        f"{prefix}score_mean": float(score.mean()) if len(frame) else float("nan"),
        f"{prefix}score_std": float(score.std(ddof=0)) if len(frame) else float("nan"),
        f"{prefix}score_min": float(score.min()) if len(frame) else float("nan"),
        f"{prefix}score_max": float(score.max()) if len(frame) else float("nan"),
        f"{prefix}score_p50": float(score.quantile(0.50)) if len(frame) else float("nan"),
        f"{prefix}score_p90": float(score.quantile(0.90)) if len(frame) else float("nan"),
        f"{prefix}score_p95": float(score.quantile(0.95)) if len(frame) else float("nan"),
        f"{prefix}score_p99": float(score.quantile(0.99)) if len(frame) else float("nan"),
    }
    if y_true.nunique() >= 2:
        output[f"{prefix}roc_auc_from_score"] = float(roc_auc_score_local(y_true, score))
        output[f"{prefix}pr_auc_from_score"] = float(average_precision_score_local(y_true, score))
    else:
        output[f"{prefix}roc_auc_from_score"] = float("nan")
        output[f"{prefix}pr_auc_from_score"] = float("nan")
    return output


def summarize_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_columns = [column for column in frame.columns if column.startswith("true_")]
    for event_column in event_columns:
        event_mask = frame[event_column].fillna(0).astype(int) > 0
        normal_mask = ~event_mask
        row = {
            "event_type": event_column.replace("true_", ""),
            "event_windows": int(event_mask.sum()),
            "non_event_windows": int(normal_mask.sum()),
        }
        if event_mask.any():
            row.update(
                {
                    "event_score_mean": float(frame.loc[event_mask, "score"].mean()),
                    "event_score_p95": float(frame.loc[event_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"event_score_mean": float("nan"), "event_score_p95": float("nan")})
        if normal_mask.any():
            row.update(
                {
                    "non_event_score_mean": float(frame.loc[normal_mask, "score"].mean()),
                    "non_event_score_p95": float(frame.loc[normal_mask, "score"].quantile(0.95)),
                }
            )
        else:
            row.update({"non_event_score_mean": float("nan"), "non_event_score_p95": float("nan")})
        rows.append(row)
    return pd.DataFrame(rows)


def _curve_points(y_true: pd.Series, y_score: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.DataFrame({"y": y_true.astype(int), "score": y_score.astype(float)}).dropna()
    if data.empty or data["y"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    data = data.sort_values("score", ascending=False).reset_index(drop=True)
    positives = float((data["y"] == 1).sum())
    negatives = float((data["y"] == 0).sum())
    tp = data["y"].cumsum()
    fp = np.arange(1, len(data) + 1) - tp
    roc = pd.DataFrame({"fpr": fp / negatives, "tpr": tp / positives})
    roc = pd.concat([pd.DataFrame({"fpr": [0.0], "tpr": [0.0]}), roc], ignore_index=True)
    pr = pd.DataFrame({"recall": tp / positives, "precision": tp / np.arange(1, len(data) + 1)})
    pr = pd.concat([pd.DataFrame({"recall": [0.0], "precision": [1.0]}), pr], ignore_index=True)
    return roc, pr


def _plot_curve(frame: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.plot(frame[x], frame[y], linewidth=1.5)
    if x == "fpr":
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel(x.upper() if x == "fpr" else x.title())
    ax.set_ylabel(y.upper() if y == "tpr" else y.title())
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def write_evaluation_report(test_df: pd.DataFrame, reports_dir: Path, metrics: dict[str, object]) -> None:
    y_true = test_df["y_true"].astype(int)
    score = test_df["score"].astype(float)
    roc, pr = _curve_points(y_true, score)
    figures_dir = reports_dir / "figures"
    if not roc.empty:
        roc.to_csv(reports_dir / "roc_curve.csv", index=False)
        _plot_curve(roc, "fpr", "tpr", "ROC Curve", figures_dir / "roc_curve.png")
    if not pr.empty:
        pr.to_csv(reports_dir / "precision_recall_curve.csv", index=False)
        _plot_curve(pr, "recall", "precision", "Precision-Recall Curve", figures_dir / "precision_recall_curve.png")

    validation_scores = reports_dir / "validation_scores.csv"
    threshold_note = "threshold metrics skipped: validation_scores.csv not found"
    threshold_metrics: dict[str, object] = {}
    if validation_scores.exists():
        val_df = pd.read_csv(validation_scores)
        if "score" in val_df.columns:
            threshold = float(pd.to_numeric(val_df["score"], errors="coerce").quantile(0.95))
            threshold_metrics = binary_metrics(y_true, score, threshold=threshold)
            threshold_note = f"threshold = validation score p95 = {threshold:.6g}"
            pd.DataFrame([{**{"threshold": threshold}, **threshold_metrics}]).to_csv(
                reports_dir / "threshold_metrics.csv",
                index=False,
            )

    component_rows = []
    for column in ["score", "nll", "reconstruction_error", "divergence", "association_discrepancy", "tail_z_abs"]:
        if column in test_df.columns:
            values = pd.to_numeric(test_df[column], errors="coerce")
            component_rows.append(
                {
                    "metric": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "p95": float(values.quantile(0.95)),
                    "roc_auc": float(roc_auc_score_local(y_true, values)) if y_true.nunique() >= 2 else float("nan"),
                    "pr_auc": float(average_precision_score_local(y_true, values)) if y_true.nunique() >= 2 else float("nan"),
                }
            )
    pd.DataFrame(component_rows).to_csv(reports_dir / "component_metrics.csv", index=False)

    lines = [
        "# Test Evaluation Report",
        "",
        "## What was measured",
        "",
        "- ROC-AUC: ranks anomaly windows above normal windows across all thresholds.",
        "- PR-AUC: precision/recall area, more informative when anomaly labels are rare or dense.",
        "- Threshold metrics: precision, sensitivity, specificity, F1, TP/TN/FP/FN using validation-only p95 threshold when available.",
        "- Component metrics: same ROC/PR check for raw score components, so weak parts are visible instead of hidden in the final score.",
        "",
        "## Result",
        "",
        f"- windows: {int(metrics.get('n_windows', len(test_df)))}",
        f"- anomaly rate: {float(metrics.get('anomaly_rate', float('nan'))):.6f}",
        f"- ROC-AUC: {float(metrics.get('roc_auc_from_score', float('nan'))):.6f}",
        f"- PR-AUC: {float(metrics.get('pr_auc_from_score', float('nan'))):.6f}",
        f"- {threshold_note}",
    ]
    if threshold_metrics:
        lines.extend(
            [
                f"- F1: {float(threshold_metrics['f1_score']):.6f}",
                f"- sensitivity: {float(threshold_metrics['sensitivity']):.6f}",
                f"- specificity: {float(threshold_metrics['specificity']):.6f}",
                f"- precision: {float(threshold_metrics['precision']):.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `test_scores.csv`: raw window scores and labels.",
            "- `score_metrics.json`: scalar score summary.",
            "- `component_metrics.csv`: per-component ROC/PR and distribution stats.",
            "- `roc_curve.csv` / `figures/roc_curve.png`: ROC evidence.",
            "- `precision_recall_curve.csv` / `figures/precision_recall_curve.png`: PR evidence.",
        ]
    )
    (reports_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_attention_score_csv(run_dir: Path, split: str) -> Path:
    if split == "val":
        return run_dir / "reports" / "validation_scores.csv"
    if split == "test_preview":
        return run_dir / "reports" / "test_scores_preview.csv"
    return run_dir / "reports" / "test_scores.csv"


def _parse_optional_ints(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(item.strip()) for item in str(value).split(",") if item.strip()}


def _parse_optional_strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _robust_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    clean = numeric.dropna()
    if clean.empty:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - median) / scale


def _softmax_negative(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    clean = numeric.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    shifted = -clean
    shifted = shifted - shifted.max(skipna=True)
    exp_values = np.exp(shifted.fillna(-np.inf))
    total = float(exp_values.sum())
    if total <= 0 or not np.isfinite(total):
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return pd.Series(exp_values / total, index=numeric.index)


def add_robust_nll_softmax_assoc_score(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "nll", "association_discrepancy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Robust score requires columns: {', '.join(missing)}")
    out = frame.copy()
    out["nll"] = pd.to_numeric(out["nll"], errors="coerce")
    out["association_discrepancy"] = pd.to_numeric(out["association_discrepancy"], errors="coerce")
    out["nll_robust_z"] = out.groupby("ticker", group_keys=False)["nll"].apply(_robust_z)
    out["association_robust_z"] = out.groupby("ticker", group_keys=False)["association_discrepancy"].apply(_robust_z)
    out["association_softmax_neg_robust_z"] = out.groupby("ticker", group_keys=False)["association_robust_z"].apply(
        _softmax_negative
    )
    out["nll_robust_softmax_assoc_score"] = out["nll_robust_z"] * out["association_softmax_neg_robust_z"]
    return out


def _mad_threshold(values: pd.Series, k: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return float(clean.mean() + k * clean.std(ddof=0))
    return median + float(k) * scale


def add_endpoint_log_return_labels(results: pd.DataFrame, data_path: Path) -> pd.DataFrame:
    required = {"ticker", "end_date"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Endpoint log-return labels require columns: {', '.join(missing)}")

    frames = []
    for ticker, frame in results.groupby("ticker", sort=False):
        frame = frame.copy()
        frame["end_date"] = pd.to_datetime(frame["end_date"])
        ohlcv_path = data_path / f"{ticker}_ohlcv.csv"
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"Missing OHLCV file for endpoint log-return labels: {ohlcv_path}")
        ohlcv = pd.read_csv(ohlcv_path)
        if "Date" not in ohlcv.columns:
            ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "Date"})
        if "Close" not in ohlcv.columns:
            raise ValueError(f"{ohlcv_path} is missing Close")
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
        ohlcv = ohlcv.sort_values("Date").reset_index(drop=True)
        ohlcv["log_return"] = np.log(
            pd.to_numeric(ohlcv["Close"], errors="coerce")
            / pd.to_numeric(ohlcv["Close"], errors="coerce").shift(1)
        )

        endpoint_dates = set(pd.to_datetime(frame["end_date"]))
        endpoints = ohlcv[ohlcv["Date"].isin(endpoint_dates)].copy()
        log_return = pd.to_numeric(endpoints["log_return"], errors="coerce")
        mean_lr = float(log_return.mean()) if log_return.notna().any() else float("nan")
        std_lr = float(log_return.std(ddof=0)) if log_return.notna().any() else float("nan")
        upper = mean_lr + 3.0 * std_lr if np.isfinite(std_lr) else float("nan")
        lower = mean_lr - 3.0 * std_lr if np.isfinite(std_lr) else float("nan")
        endpoints["endpoint_log_return_anomaly"] = (
            (pd.to_numeric(endpoints["log_return"], errors="coerce") > upper)
            | (pd.to_numeric(endpoints["log_return"], errors="coerce") < lower)
        ).astype(int)
        endpoints["endpoint_log_return_mean"] = mean_lr
        endpoints["endpoint_log_return_std"] = std_lr
        endpoints["endpoint_log_return_upper_3std"] = upper
        endpoints["endpoint_log_return_lower_3std"] = lower

        merged = frame.merge(
            endpoints[
                [
                    "Date",
                    "log_return",
                    "endpoint_log_return_anomaly",
                    "endpoint_log_return_mean",
                    "endpoint_log_return_std",
                    "endpoint_log_return_upper_3std",
                    "endpoint_log_return_lower_3std",
                ]
            ],
            left_on="end_date",
            right_on="Date",
            how="left",
        ).drop(columns=["Date"])
        merged = merged.rename(columns={"log_return": "endpoint_log_return"})
        merged["endpoint_log_return_anomaly"] = merged["endpoint_log_return_anomaly"].fillna(0).astype(int)
        merged["true_log_return_anomaly"] = merged["endpoint_log_return_anomaly"]
        merged["y_true"] = merged["endpoint_log_return_anomaly"]
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def _price_anomaly_start_dates(ticker: str, price_dir: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, z_thr: float) -> set[pd.Timestamp]:
    price_path = price_dir / f"{ticker}_ohlcv.csv"
    if not price_path.exists():
        return set()
    price_df = pd.read_csv(price_path, parse_dates=["Date"])
    price_df = price_df[(price_df["Date"] >= start_date) & (price_df["Date"] <= end_date)].copy()
    if price_df.empty or "Close" not in price_df.columns:
        return set()
    price_df = price_df.sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(price_df["Close"], errors="coerce")
    returns = close.pct_change()
    ret_z = _robust_z(returns)
    is_anomaly = ret_z.abs() > float(z_thr)
    starts: set[pd.Timestamp] = set()
    previous = False
    for date, current in zip(pd.to_datetime(price_df["Date"]), is_anomaly.fillna(False)):
        current_bool = bool(current)
        if current_bool and not previous:
            starts.add(pd.to_datetime(date))
        previous = current_bool
    return starts


def add_price_overlap_columns(frame: pd.DataFrame, price_dir: Path, price_z_thr: float) -> pd.DataFrame:
    required = {"ticker", "end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Overlap filtering requires columns: {', '.join(missing)}")
    out = frame.copy()
    out["end_date"] = pd.to_datetime(out["end_date"])
    out["is_price_anom_start"] = False
    for ticker, group in out.groupby("ticker", sort=False):
        start_date = pd.to_datetime(group["end_date"]).min()
        end_date = pd.to_datetime(group["end_date"]).max()
        starts = _price_anomaly_start_dates(str(ticker), price_dir, start_date, end_date, price_z_thr)
        if starts:
            out.loc[group.index, "is_price_anom_start"] = group["end_date"].isin(starts).to_numpy()
    out["is_delta_price_overlap"] = out["is_delta_mad_spike"].fillna(False) & out["is_price_anom_start"].fillna(False)
    return out


def _attention_top_k(value, default: int = 5) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and value.lower() in {"all", "none", "null", "full"}:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _limit_per_ticker(frame: pd.DataFrame, per_ticker) -> pd.DataFrame:
    if per_ticker is None:
        return frame
    limit = int(per_ticker)
    if limit <= 0 or "ticker" not in frame.columns:
        return frame
    return frame.groupby("ticker", group_keys=False, sort=False).head(limit).reset_index(drop=True)


def _select_extreme_and_normal_context(frame: pd.DataFrame, args) -> pd.DataFrame:
    data_path = Path(getattr(args, "data_path", ""))
    if not data_path:
        raise ValueError("endpoint_log_return_context selection requires data_path")
    if "endpoint_log_return" not in frame.columns:
        frame = add_endpoint_log_return_labels(frame, data_path)
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["endpoint_log_return"] = pd.to_numeric(frame["endpoint_log_return"], errors="coerce")
    frame["abs_endpoint_log_return"] = frame["endpoint_log_return"].abs()
    frame = frame.dropna(subset=["endpoint_log_return", "abs_endpoint_log_return"]).copy()
    if frame.empty:
        raise ValueError("No endpoint log-return rows available")

    reference_count = int(getattr(args, "attention_reference_count", 1) or 1)
    context_days = int(getattr(args, "attention_context_days", 10) or 10)
    mean_return = float(frame["endpoint_log_return"].mean())
    frame["endpoint_log_return_mean_gap"] = (frame["endpoint_log_return"] - mean_return).abs()

    extreme_refs = frame.sort_values("abs_endpoint_log_return", ascending=False).head(reference_count).copy()
    extreme_refs["reference_role"] = "max_abs_log_return"
    normal_pool = frame[pd.to_numeric(frame.get("endpoint_log_return_anomaly", 0), errors="coerce").fillna(0).astype(int) == 0].copy()
    if normal_pool.empty:
        normal_pool = frame
    normal_refs = normal_pool.sort_values("endpoint_log_return_mean_gap", ascending=True).head(reference_count).copy()
    normal_refs["reference_role"] = "normal_near_mean"

    ordered = frame.sort_values(["ticker", "end_date", "window_id"]).reset_index(drop=True)
    selected_rows = []
    for reference in pd.concat([extreme_refs, normal_refs], ignore_index=True).itertuples(index=False):
        matches = ordered.index[ordered["window_id"].astype(int) == int(reference.window_id)].tolist()
        if not matches:
            continue
        ref_pos = int(matches[0])
        for lag in range(context_days, -1, -1):
            pos = ref_pos - lag
            if pos < 0 or str(ordered.loc[pos, "ticker"]) != str(reference.ticker):
                continue
            row = ordered.loc[pos].copy()
            row["reference_role"] = reference.reference_role
            row["reference_window_id"] = int(reference.window_id)
            row["reference_end_date"] = reference.end_date
            row["reference_endpoint_log_return"] = float(reference.endpoint_log_return)
            row["reference_abs_endpoint_log_return"] = float(reference.abs_endpoint_log_return)
            row["lag_from_reference_days"] = int(lag)
            row["selection_role"] = f"{reference.reference_role}_lag{lag}"
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise ValueError("No context windows selected")
    selected = selected.drop_duplicates(subset=["window_id", "reference_role", "lag_from_reference_days"])
    selected = selected.sort_values(["reference_role", "reference_window_id", "lag_from_reference_days"], ascending=[True, True, False])
    selected.attrs["score_context"] = ordered
    return selected.reset_index(drop=True)


def _select_extreme_log_return_context(frame: pd.DataFrame, args) -> pd.DataFrame:
    data_path = Path(getattr(args, "data_path", ""))
    if not data_path:
        raise ValueError("extreme_log_return_context selection requires data_path")
    if "endpoint_log_return" not in frame.columns:
        frame = add_endpoint_log_return_labels(frame, data_path)
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["endpoint_log_return"] = pd.to_numeric(frame["endpoint_log_return"], errors="coerce")
    frame = frame.dropna(subset=["endpoint_log_return"]).copy()
    if frame.empty:
        raise ValueError("No endpoint log-return rows available")

    threshold_std = float(getattr(args, "attention_log_return_std_threshold", 3.0) or 3.0)
    reference_count = int(getattr(args, "attention_reference_count", 12) or 12)
    context_days = int(getattr(args, "attention_context_days", 6) or 6)
    rows = []
    for ticker, group in frame.groupby("ticker", sort=False):
        returns = pd.to_numeric(group["endpoint_log_return"], errors="coerce")
        mean = float(returns.mean())
        std = float(returns.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            continue
        local = group.copy()
        local["endpoint_log_return_z"] = (local["endpoint_log_return"] - mean) / std
        rows.append(local[local["endpoint_log_return_z"].abs() >= threshold_std])
    if not rows:
        raise ValueError(f"No endpoint log-return rows exceed +/-{threshold_std:g} std")

    references = pd.concat(rows, ignore_index=True)
    references["reference_role"] = "extreme_log_return"
    references = references.sort_values("endpoint_log_return_z", key=lambda values: values.abs(), ascending=False).head(reference_count)

    ordered = frame.sort_values(["ticker", "end_date", "window_id"]).reset_index(drop=True)
    selected_rows = []
    for reference in references.itertuples(index=False):
        matches = ordered.index[ordered["window_id"].astype(int) == int(reference.window_id)].tolist()
        if not matches:
            continue
        ref_pos = int(matches[0])
        for lag in range(context_days, -1, -1):
            pos = ref_pos - lag
            if pos < 0 or str(ordered.loc[pos, "ticker"]) != str(reference.ticker):
                continue
            row = ordered.loc[pos].copy()
            row["reference_role"] = "extreme_log_return"
            row["reference_window_id"] = int(reference.window_id)
            row["reference_end_date"] = reference.end_date
            row["reference_endpoint_log_return"] = float(reference.endpoint_log_return)
            row["reference_abs_endpoint_log_return"] = float(abs(reference.endpoint_log_return))
            row["reference_endpoint_log_return_z"] = float(reference.endpoint_log_return_z)
            row["lag_from_reference_days"] = int(lag)
            row["selection_role"] = f"extreme_log_return_lag{lag}"
            selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise ValueError("No extreme context windows selected")
    selected = selected.drop_duplicates(subset=["window_id", "reference_window_id", "lag_from_reference_days"])
    selected = selected.sort_values(["reference_window_id", "lag_from_reference_days"], ascending=[True, False])
    selected.attrs["score_context"] = ordered
    return selected.reset_index(drop=True)


def _add_endpoint_daily_labels(frame: pd.DataFrame, data_path: Path, label_columns: list[str]) -> pd.DataFrame:
    required = {"ticker", "end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Endpoint label selection requires columns: {', '.join(missing)}")
    out = frame.copy()
    out["end_date"] = pd.to_datetime(out["end_date"])
    for label_column in label_columns:
        out[f"endpoint_{label_column}"] = 0

    for ticker, group in out.groupby("ticker", sort=False):
        label_path = data_path / f"{ticker}_anomaly_label.csv"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing daily label file for endpoint selection: {label_path}")
        labels = pd.read_csv(label_path)
        if "Date" not in labels.columns:
            labels = labels.rename(columns={labels.columns[0]: "Date"})
        labels["Date"] = pd.to_datetime(labels["Date"])
        available = [column for column in label_columns if column in labels.columns]
        if not available:
            continue
        labels = labels[["Date", *available]].copy()
        merged = group[["end_date"]].merge(labels, left_on="end_date", right_on="Date", how="left")
        for label_column in available:
            out.loc[group.index, f"endpoint_{label_column}"] = (
                pd.to_numeric(merged[label_column], errors="coerce").fillna(0).astype(int).to_numpy()
            )
    return out


def _combine_score_terms(nll: pd.Series, association: pd.Series, formula: str) -> pd.Series:
    if formula == "sum":
        return nll.fillna(0.0) + association.fillna(0.0)
    if formula == "product":
        return nll.fillna(0.0) * association.fillna(0.0)
    raise ValueError("attention_score_formula must be sum, product, or softmax_product")


def add_delta_attention_selection_columns(frame: pd.DataFrame, mad_k: float = 10.0, score_formula: str = "sum") -> pd.DataFrame:
    required = {"ticker", "end_date", "window_id", "nll", "association_discrepancy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Delta attention selection requires columns: {', '.join(missing)}")

    out = frame.copy()
    out["end_date"] = pd.to_datetime(out["end_date"])
    out["nll"] = pd.to_numeric(out["nll"], errors="coerce")
    out["association_discrepancy"] = pd.to_numeric(out["association_discrepancy"], errors="coerce")
    score_formula = str(score_formula or "sum").lower()
    out["score_formula"] = score_formula
    if score_formula in {"softmax_product", "nll_assoc_softmax_product"}:
        if "softmax_product_score" not in out.columns:
            raise ValueError("attention_score_formula=softmax_product requires softmax_product_score in the score CSV")
        out["raw_sum"] = pd.to_numeric(out["softmax_product_score"], errors="coerce")
        out["score_formula"] = "softmax_product"
    elif score_formula in {"mle_param_softmax_product", "mle_param"}:
        if "mle_param_softmax_product_score" not in out.columns:
            raise ValueError("attention_score_formula=mle_param_softmax_product requires mle_param_softmax_product_score in the score CSV")
        out["raw_sum"] = pd.to_numeric(out["mle_param_softmax_product_score"], errors="coerce")
        out["score_formula"] = "mle_param_softmax_product"
    else:
        out["raw_sum"] = _combine_score_terms(out["nll"], out["association_discrepancy"], score_formula)
    out = out.sort_values(["ticker", "end_date", "window_id"]).reset_index(drop=True)
    out["score_robust_z"] = out.groupby("ticker", group_keys=False)["raw_sum"].apply(_robust_z)
    out["prev_score_robust_z"] = out.groupby("ticker")["score_robust_z"].shift(1)
    out["delta_score_robust_z"] = out.groupby("ticker")["score_robust_z"].diff().abs()
    out["delta_mad_threshold"] = out.groupby("ticker")["delta_score_robust_z"].transform(
        lambda values: _mad_threshold(values, mad_k)
    )
    out["delta_mad_k"] = float(mad_k)
    out["is_delta_mad_spike"] = out["delta_score_robust_z"] > out["delta_mad_threshold"]
    return out


def select_attention_windows(args, run_dir: Path) -> pd.DataFrame:
    split = str(getattr(args, "attention_split", "test") or "test")
    score_csv_value = getattr(args, "attention_score_csv", None)
    score_csv = Path(score_csv_value) if score_csv_value else _default_attention_score_csv(run_dir, split)
    if not score_csv.exists():
        raise FileNotFoundError(f"Missing score CSV for attention selection: {score_csv}")

    frame = pd.read_csv(score_csv)
    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == split].copy()
    attention_tickers = _parse_optional_strings(getattr(args, "attention_tickers", None))
    if attention_tickers:
        frame = frame[frame["ticker"].astype(str).isin(attention_tickers)].copy()
    elif getattr(args, "tickers", None):
        frame = frame[frame["ticker"].astype(str).isin([str(ticker) for ticker in args.tickers])].copy()

    manual_window_ids = _parse_optional_ints(getattr(args, "attention_window_ids", None))
    if manual_window_ids:
        selected = frame[frame["window_id"].astype(int).isin(manual_window_ids)].copy()
    else:
        select_by = str(getattr(args, "attention_select_by", "association_discrepancy") or "association_discrepancy")
        if select_by in {
            "delta",
            "delta_score",
            "delta_score_robust_z",
            "delta_mad",
            "delta_mad_spike",
            "delta_mad_previous",
            "delta_mad_spike_previous",
            "delta_low",
            "delta_low_normal",
        }:
            frame = add_delta_attention_selection_columns(
                frame,
                mad_k=float(getattr(args, "attention_mad_k", 10.0)),
                score_formula=str(getattr(args, "attention_score_formula", "sum") or "sum"),
            )
        if select_by in {"delta", "delta_score"}:
            select_by = "delta_score_robust_z"
        if select_by in {"delta_low", "delta_low_normal"}:
            price_dir = Path(getattr(args, "price_dir", "datasets/SP500") or "datasets/SP500")
            price_z_thr = float(getattr(args, "price_z_thr", 3.0) or 3.0)
            frame = add_price_overlap_columns(frame, price_dir=price_dir, price_z_thr=price_z_thr)
            selected = frame[
                (~frame["is_delta_mad_spike"].fillna(False))
                & (~frame["is_price_anom_start"].fillna(False))
                & frame["delta_score_robust_z"].notna()
            ].copy()
            selected = selected.sort_values(["ticker", "delta_score_robust_z", "end_date"], ascending=[True, True, True])
            selected = _limit_per_ticker(selected, getattr(args, "attention_per_ticker", None))
            sort_column = "delta_score_robust_z"
            top_k = _attention_top_k(getattr(args, "attention_top_k", 5), default=5)
            selected = selected.sort_values(sort_column, ascending=True)
            if top_k is not None:
                selected = selected.head(top_k)
            if selected.empty:
                raise ValueError("No attention windows selected")
            result = selected.reset_index(drop=True)
            result.attrs["score_context"] = frame
            return result
        if select_by in {"endpoint_log_return_context", "max_abs_log_return_context", "extreme_normal_context"}:
            selected = _select_extreme_and_normal_context(frame, args)
            selected.attrs["score_context"] = selected.attrs.get("score_context")
            return selected
        if select_by in {"extreme_log_return_context", "extreme_return_context"}:
            selected = _select_extreme_log_return_context(frame, args)
            selected.attrs["score_context"] = selected.attrs.get("score_context")
            return selected
        if select_by in {"delta_mad", "delta_mad_spike", "delta_mad_previous", "delta_mad_spike_previous"}:
            if getattr(args, "attention_overlap_only", False):
                price_dir = Path(getattr(args, "price_dir", "datasets/SP500") or "datasets/SP500")
                price_z_thr = float(getattr(args, "price_z_thr", 3.0) or 3.0)
                frame = add_price_overlap_columns(frame, price_dir=price_dir, price_z_thr=price_z_thr)
                spikes = frame[frame["is_delta_price_overlap"].fillna(False)].copy()
            else:
                spikes = frame[frame["is_delta_mad_spike"].fillna(False)].copy()
            if select_by in {"delta_mad_previous", "delta_mad_spike_previous"}:
                previous_rows = []
                for spike_idx, spike_row in spikes.iterrows():
                    prev_idx = int(spike_idx) - 1
                    if prev_idx < 0 or str(frame.loc[prev_idx, "ticker"]) != str(spike_row["ticker"]):
                        continue
                    previous = frame.loc[prev_idx].copy()
                    previous["reference_spike_window_id"] = int(spike_row["window_id"])
                    previous["reference_spike_end_date"] = spike_row["end_date"]
                    previous["reference_spike_delta_score_robust_z"] = spike_row["delta_score_robust_z"]
                    previous["reference_spike_score_robust_z"] = spike_row["score_robust_z"]
                    previous["selection_role"] = "previous_to_delta_mad_spike"
                    previous_rows.append(previous)
                selected = pd.DataFrame(previous_rows)
                if not selected.empty:
                    selected = selected.drop_duplicates(subset=["ticker", "window_id"]).copy()
            else:
                selected = spikes
            sort_column = "delta_score_robust_z"
            top_k = _attention_top_k(getattr(args, "attention_top_k", "all"), default=0)
            if top_k is not None:
                selected = selected.sort_values(sort_column, ascending=False).head(top_k)
            if selected.empty:
                raise ValueError("No attention windows selected")
            result = selected.reset_index(drop=True)
            result.attrs["score_context"] = frame
            return result
        if select_by == "labeled":
            if "y_true" not in frame.columns:
                raise ValueError("attention_select_by=labeled requires y_true in the score CSV")
            selected = frame[frame["y_true"].astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        elif select_by in {"log_return_anomaly", "true_log_return_anomaly"}:
            event_column = "true_log_return_anomaly"
            if event_column not in frame.columns:
                raise ValueError(f"attention_select_by={select_by} requires {event_column} in the score CSV")
            selected = frame[pd.to_numeric(frame[event_column], errors="coerce").fillna(0).astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        elif select_by in {"endpoint_log_return_anomaly", "daily_log_return_anomaly"}:
            data_path = Path(getattr(args, "data_path", ""))
            if not data_path:
                raise ValueError(f"attention_select_by={select_by} requires data_path")
            frame = _add_endpoint_daily_labels(frame, data_path=data_path, label_columns=["log_return_anomaly"])
            selected = frame[frame["endpoint_log_return_anomaly"].astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        elif select_by in {"endpoint_abs_log_return", "abs_endpoint_log_return", "max_abs_log_return"}:
            data_path = Path(getattr(args, "data_path", ""))
            if not data_path:
                raise ValueError(f"attention_select_by={select_by} requires data_path")
            frame = add_endpoint_log_return_labels(frame, data_path)
            frame["abs_endpoint_log_return"] = pd.to_numeric(
                frame["endpoint_log_return"],
                errors="coerce",
            ).abs()
            selected = frame.dropna(subset=["abs_endpoint_log_return"]).copy()
            sort_column = "abs_endpoint_log_return"
        elif select_by in {"volume_anomaly", "true_volume_anomaly"}:
            event_column = "true_volume_anomaly"
            if event_column not in frame.columns:
                raise ValueError(f"attention_select_by={select_by} requires {event_column} in the score CSV")
            selected = frame[pd.to_numeric(frame[event_column], errors="coerce").fillna(0).astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        elif select_by in {"endpoint_volume_anomaly", "daily_volume_anomaly"}:
            data_path = Path(getattr(args, "data_path", ""))
            if not data_path:
                raise ValueError(f"attention_select_by={select_by} requires data_path")
            frame = _add_endpoint_daily_labels(frame, data_path=data_path, label_columns=["volume_anomaly"])
            selected = frame[frame["endpoint_volume_anomaly"].astype(int) > 0].copy()
            sort_column = "score" if "score" in selected.columns else "association_discrepancy"
        else:
            sort_column = select_by
            if sort_column not in frame.columns:
                raise ValueError(f"attention_select_by column not found in score CSV: {sort_column}")
            selected = frame.copy()
        top_k = _attention_top_k(getattr(args, "attention_top_k", 5), default=5)
        selected = selected.sort_values(sort_column, ascending=False)
        selected = _limit_per_ticker(selected, getattr(args, "attention_per_ticker", None))
        if top_k is not None:
            selected = selected.head(top_k)

    if selected.empty:
        raise ValueError("No attention windows selected")
    result = selected.reset_index(drop=True)
    result.attrs["score_context"] = frame
    return result


def _entropy(values: np.ndarray) -> float:
    safe = np.clip(values.astype(float), 1e-12, None)
    return float(-(safe * np.log(safe)).sum())


def _attention_summary(series: np.ndarray, prior: np.ndarray) -> dict[str, float | int]:
    endpoint_series = series[:, :, -1, :]
    endpoint_prior = prior[:, :, -1, :]
    endpoint_abs_diff = np.abs(endpoint_series - endpoint_prior)
    averaged_endpoint = endpoint_series.mean(axis=(0, 1))
    top_index = int(np.argmax(averaged_endpoint))
    length = int(averaged_endpoint.shape[0])
    return {
        "endpoint_l1_mean": float(endpoint_abs_diff.sum(axis=-1).mean()),
        "endpoint_series_entropy_mean": float(np.mean([_entropy(row) for row in endpoint_series.reshape(-1, length)])),
        "endpoint_prior_entropy_mean": float(np.mean([_entropy(row) for row in endpoint_prior.reshape(-1, length)])),
        "endpoint_top_index": top_index,
        "endpoint_top_lag": int(length - 1 - top_index),
        "endpoint_top_weight": float(averaged_endpoint[top_index]),
    }


def export_attention_artifacts(args) -> pd.DataFrame:
    set_seed(args.seed)
    device = resolve_device(args.device)
    split = str(getattr(args, "attention_split", "test") or "test")
    if split not in {"val", "test"}:
        raise ValueError("attention_split must be val or test")
    print(f"[attention] device={device} split={split}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    selected = select_attention_windows(args, run_dir)
    target_ids = {int(value) for value in selected["window_id"].tolist()}
    print(f"[attention] selected windows={len(target_ids)} from score CSV", flush=True)

    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    loader = val_loader if split == "val" else test_loader
    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))
    model.eval()

    out_root_value = getattr(args, "attention_out", None)
    out_root = Path(out_root_value) if out_root_value else run_dir / "reports" / "attention"
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    context_frame = selected.attrs.get("score_context")
    context_path = None
    if isinstance(context_frame, pd.DataFrame):
        context_path = out_root / "attention_score_context.csv"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_frame.to_csv(context_path, index=False)

    score_lookup = selected.set_index(selected["window_id"].astype(int)).to_dict(orient="index")
    rows = []
    with torch.no_grad():
        for batch in loader:
            meta = batch["meta"]
            batch_window_ids = [int(value) for value in meta["window_id"]]
            wanted_positions = [idx for idx, window_id in enumerate(batch_window_ids) if window_id in target_ids]
            if not wanted_positions:
                continue

            x = batch["x"].to(device)
            returns = batch["returns"].to(device)
            time_deltas = batch["time_deltas"].to(device)
            _, _, _, _, attn_maps, _, _, latent, association = model(
                x,
                returns=returns,
                time_deltas=time_deltas,
                return_attention=True,
            )
            series_stack = torch.stack([attn["series"] for attn in attn_maps], dim=1).detach().cpu().numpy()
            prior_stack = torch.stack([attn["prior"] for attn in attn_maps], dim=1).detach().cpu().numpy()
            latent_stack = latent.detach().cpu().numpy()
            scaled_x_stack = x.detach().cpu().numpy()
            returns_stack = returns.detach().cpu().numpy()
            time_deltas_stack = time_deltas.detach().cpu().numpy()
            assoc_values = association.detach().cpu().numpy() if association is not None else np.zeros(len(batch_window_ids))

            for idx in wanted_positions:
                window_id = int(batch_window_ids[idx])
                ticker = str(meta["ticker"][idx])
                start_idx = int(meta["start_idx"][idx])
                dates = window_store[ticker]["dates"][start_idx: start_idx + int(args.window_size)]
                raw_features = window_store[ticker]["x"][start_idx: start_idx + int(args.window_size)]
                series = series_stack[idx].astype(np.float32)
                prior = prior_stack[idx].astype(np.float32)
                diff = (series - prior).astype(np.float32)
                out_path = data_dir / f"{ticker}_window{window_id}.npz"
                np.savez_compressed(
                    out_path,
                    series=series,
                    prior=prior,
                    diff=diff,
                    dates=np.asarray(dates, dtype=str),
                    feature_names=np.asarray(get_feature_columns(str(args.features))),
                    raw_features=raw_features.astype(np.float32),
                    scaled_features=scaled_x_stack[idx].astype(np.float32),
                    returns=returns_stack[idx].astype(np.float32),
                    time_deltas=time_deltas_stack[idx].astype(np.float32),
                    latent=latent_stack[idx].astype(np.float32),
                    endpoint_latent=latent_stack[idx, -1].astype(np.float32),
                    mean_latent=latent_stack[idx].mean(axis=0).astype(np.float32),
                )
                score_row = score_lookup.get(window_id, {})
                row = {
                    "ticker": ticker,
                    "split": split,
                    "window_id": window_id,
                    "start_idx": start_idx,
                    "end_idx": int(meta["end_idx"][idx]),
                    "start_date": meta["start_date"][idx],
                    "end_date": meta["end_date"][idx],
                    "y_true": int(batch["y"][idx].item()),
                    "association_discrepancy": float(assoc_values[idx]),
                    "score": float(score_row.get("score", np.nan)),
                    "nll": float(score_row.get("nll", np.nan)),
                    "raw_sum": float(score_row.get("raw_sum", np.nan)),
                    "score_formula": str(score_row.get("score_formula", getattr(args, "attention_score_formula", "sum"))),
                    "score_robust_z": float(score_row.get("score_robust_z", np.nan)),
                    "prev_score_robust_z": float(score_row.get("prev_score_robust_z", np.nan)),
                    "delta_score_robust_z": float(score_row.get("delta_score_robust_z", np.nan)),
                    "delta_mad_threshold": float(score_row.get("delta_mad_threshold", np.nan)),
                    "delta_mad_k": float(score_row.get("delta_mad_k", np.nan)),
                    "is_price_anom_start": bool(score_row.get("is_price_anom_start", False)),
                    "is_delta_price_overlap": bool(score_row.get("is_delta_price_overlap", False)),
                    "selection_role": str(score_row.get("selection_role", "selected")),
                    "reference_role": str(score_row.get("reference_role", "")),
                    "reference_window_id": int(score_row.get("reference_window_id", -1)) if pd.notna(score_row.get("reference_window_id", np.nan)) else -1,
                    "reference_end_date": str(score_row.get("reference_end_date", "")),
                    "reference_endpoint_log_return": float(score_row.get("reference_endpoint_log_return", np.nan)),
                    "reference_abs_endpoint_log_return": float(score_row.get("reference_abs_endpoint_log_return", np.nan)),
                    "reference_endpoint_log_return_z": float(score_row.get("reference_endpoint_log_return_z", np.nan)),
                    "lag_from_reference_days": int(score_row.get("lag_from_reference_days", -1)) if pd.notna(score_row.get("lag_from_reference_days", np.nan)) else -1,
                    "reference_spike_window_id": int(score_row.get("reference_spike_window_id", -1)) if pd.notna(score_row.get("reference_spike_window_id", np.nan)) else -1,
                    "reference_spike_end_date": str(score_row.get("reference_spike_end_date", "")),
                    "reference_spike_delta_score_robust_z": float(score_row.get("reference_spike_delta_score_robust_z", np.nan)),
                    "reference_spike_score_robust_z": float(score_row.get("reference_spike_score_robust_z", np.nan)),
                    "association_mode": args.association_mode,
                    "predictive_distribution": args.predictive_distribution,
                    "attention_npz": str(out_path),
                    "score_context_csv": str(context_path) if context_path is not None else "",
                }
                for column, value in score_row.items():
                    if str(column).startswith("true_"):
                        row[str(column)] = int(value) if pd.notna(value) else 0
                row.update(_attention_summary(series, prior))
                rows.append(row)
                target_ids.remove(window_id)

            if not target_ids:
                break

    if target_ids:
        print(f"[attention] warning: {len(target_ids)} selected windows were not found in {split} loader", flush=True)
    manifest_df = pd.DataFrame(rows)
    manifest_path = out_root / "attention_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[attention] saved manifest={manifest_path}", flush=True)

    if getattr(args, "attention_make_plots", True):
        from src.gbm.visualize import run_attention_visualize

        plot_args = type(
            "AttentionPlotArgs",
            (),
            {
                "attention_manifest": str(manifest_path),
                "out": str(out_root / "figures"),
                "attention_layer": getattr(args, "attention_layer", 0),
                "attention_head": getattr(args, "attention_head", 0),
                "attention_mask_diagonal": getattr(args, "attention_mask_diagonal", False),
                "price_dir": getattr(args, "price_dir", "datasets/SP500"),
            },
        )()
        run_attention_visualize(plot_args)
    return manifest_df


def validate_model(args) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[validate] device={device}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[validate] loading checkpoint={model_path}")
    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[validate] windows total={len(manifest)} | val={len(val_ds)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    val_df = collect_joint_scores(model, val_loader, device, phase="validate", **score_kwargs(args))
    test_df = collect_joint_scores(model, test_loader, device, phase="test_preview", **score_kwargs(args))
    if val_df.empty:
        raise RuntimeError("Validation scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dist_weight": args.dist_weight,
        "recon_weight": args.recon_weight,
        "divergence_weight": args.divergence_weight,
        "association_weight": args.association_weight,
        "score_mode": args.score_mode,
        "quantile_count": args.quantile_count,
        "tail_weight_gamma": args.tail_weight_gamma,
        "tail_weight_power": args.tail_weight_power,
        "predictive_distribution": args.predictive_distribution,
        "association_mode": args.association_mode,
        "val_score_mean": float(val_df["score"].mean()),
        "val_score_std": float(val_df["score"].std(ddof=0)),
        "test_preview_score_mean": float(test_df["score"].mean()),
        "test_preview_score_std": float(test_df["score"].std(ddof=0)),
        "val_windows": int(len(val_df)),
        "test_preview_windows": int(len(test_df)),
        "ticker_count": int(manifest["ticker"].nunique()),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }
    save_json(reports_dir / "score_summary.json", summary)
    val_df.to_csv(reports_dir / "validation_scores.csv", index=False)
    test_df.to_csv(reports_dir / "test_scores_preview.csv", index=False)
    print(f"Saved validation summary to {reports_dir / 'score_summary.json'}")
    return summary


def test_model(args) -> dict[str, object]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[test] device={device}", flush=True)

    run_dir = get_run_dir(args.exp_name, getattr(args, "output_root", None))
    model_path = checkpoint_path(args, filename="gbm.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    print(f"[test] loading checkpoint={model_path}")
    manifest, window_store, scaler, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, input_dim = build_runtime_loaders(args)
    print(f"[test] windows total={len(manifest)} | test={len(test_ds)} | tickers={manifest['ticker'].nunique()}")

    model = build_gbm_model(args, input_dim, device)
    model.load_state_dict(load_state_dict(model_path, device))

    test_df = collect_joint_scores(model, test_loader, device, phase="test", **score_kwargs(args))
    if test_df.empty:
        raise RuntimeError("Test scoring returned no rows")

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = reports_dir / "test_scores.csv"

    if getattr(args, "fast_export", False):
        test_df.to_csv(scores_path, index=False)
        print(f"Saved raw test scores to {scores_path}")
        return {"scores": str(scores_path)}

    label_mode = str(getattr(args, "test_label_mode", "") or getattr(args, "label_mode", "") or "").lower()
    if label_mode in {"endpoint_log_return_3std", "endpoint_log_return_anomaly", "log_return_3std"}:
        test_df = add_endpoint_log_return_labels(test_df, Path(args.data_path))
    else:
        test_df = add_test_event_columns(test_df, Path(args.data_path))
    test_df = add_robust_nll_softmax_assoc_score(test_df)
    metrics = score_summary(test_df)
    metrics.update(
        {
            "predictive_distribution": args.predictive_distribution,
            "association_mode": args.association_mode,
            "score_mode": args.score_mode,
            "ticker_count": int(manifest["ticker"].nunique()),
            "mean_reconstruction_error": float(test_df["reconstruction_error"].mean()),
            "mean_nll": float(test_df["nll"].mean()),
            "mean_divergence": float(test_df["divergence"].mean()),
            "mean_association_discrepancy": float(test_df["association_discrepancy"].mean()),
        }
    )

    metrics_path = reports_dir / "score_metrics.json"
    test_df.to_csv(scores_path, index=False)
    save_json(metrics_path, metrics)

    by_ticker_dir = reports_dir / "by_ticker"
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = []
    for ticker, group in test_df.groupby("ticker"):
        group.to_csv(by_ticker_dir / f"{ticker}_test_scores.csv", index=False)
        row = {"ticker": ticker}
        row.update(score_summary(group))
        ticker_rows.append(row)
    pd.DataFrame(ticker_rows).to_csv(reports_dir / "score_metrics_by_ticker.csv", index=False)

    event_summary = summarize_events(test_df)
    if not event_summary.empty:
        event_summary.to_csv(reports_dir / "score_summary_by_event_type.csv", index=False)
    write_evaluation_report(test_df, reports_dir, metrics)

    print(f"Saved raw test scores to {scores_path}")
    print(f"Saved raw score metrics to {metrics_path}")
    return metrics


__all__ = [
    "build_gbm_model",
    "build_runtime_loaders",
    "checkpoint_path",
    "collect_joint_scores",
    "add_delta_attention_selection_columns",
    "add_endpoint_log_return_labels",
    "add_price_overlap_columns",
    "add_robust_nll_softmax_assoc_score",
    "export_attention_artifacts",
    "load_state_dict",
    "score_kwargs",
    "score_summary",
    "select_attention_windows",
    "summarize_events",
    "test_model",
    "validate_model",
]
