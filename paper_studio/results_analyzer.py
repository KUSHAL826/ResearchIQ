"""Studio Stage 9: Real CSV/Excel Experimental Results Ingestion & Quantitative Analysis."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from core.project_manager import get_project_temp_dir, sanitize_filename
from core.schemas import ResearchProject, UploadedDatasetInfo


def load_dataframe_from_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV or Excel bytes into a pandas DataFrame."""
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw_bytes))
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    raise ValueError("Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")


def register_uploaded_dataset(
    project: ResearchProject,
    raw_bytes: bytes,
    filename: str,
) -> Tuple[UploadedDatasetInfo, pd.DataFrame]:
    """Validate, store in project temp directory, and compute real numeric summary stats."""
    df = load_dataframe_from_bytes(raw_bytes, filename)
    if df.empty:
        raise ValueError(f"Uploaded dataset '{filename}' contains zero rows.")

    safe_name = sanitize_filename(filename)
    ds_info = UploadedDatasetInfo(
        filename=filename,
        row_count=int(len(df)),
        columns=[str(c) for c in df.columns],
    )
    relpath = f"assets/datasets/{ds_info.dataset_id}_{safe_name}"
    ds_info.asset_relpath = relpath

    temp_dir = get_project_temp_dir(project.project_id)
    (temp_dir / "datasets" / Path(relpath).name).write_bytes(raw_bytes)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_summary: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if not series.empty:
            numeric_summary[str(col)] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std(ddof=0)), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }
    ds_info.numeric_summary = numeric_summary
    ds_info.preview_rows = df.head(20).fillna("").to_dict(orient="records")
    return ds_info, df


def load_project_dataset_df(project: ResearchProject, ds: UploadedDatasetInfo, asset_bytes: dict[str, bytes] | None = None) -> pd.DataFrame | None:
    """Reload DataFrame from session asset cache or project temp directory."""
    if asset_bytes and ds.asset_relpath in asset_bytes:
        return load_dataframe_from_bytes(asset_bytes[ds.asset_relpath], ds.filename)
    temp_dir = get_project_temp_dir(project.project_id)
    candidate = temp_dir / "datasets" / Path(ds.asset_relpath).name
    if candidate.exists():
        return load_dataframe_from_bytes(candidate.read_bytes(), ds.filename)
    if ds.preview_rows:
        return pd.DataFrame(ds.preview_rows)
    return None


def compute_classification_evaluation(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
) -> dict[str, Any]:
    """Compute real accuracy, precision, recall, F1, and confusion matrix from actual columns."""
    sub = df[[y_true_col, y_pred_col]].dropna()
    if sub.empty:
        raise ValueError("Selected ground-truth and prediction columns contain no valid rows.")

    y_true = sub[y_true_col].astype(str).tolist()
    y_pred = sub[y_pred_col].astype(str).tolist()
    labels = sorted(set(y_true) | set(y_pred))
    label_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    n = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    cm = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        cm[label_idx[yt]][label_idx[yp]] += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for i in range(len(labels)):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    metrics = {
        "samples": n,
        "accuracy": round(correct / max(1, n), 4),
        "precision_macro": round(sum(precisions) / max(1, len(precisions)), 4),
        "recall_macro": round(sum(recalls) / max(1, len(recalls)), 4),
        "f1_macro": round(sum(f1s) / max(1, len(f1s)), 4),
    }
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "labels": labels,
    }
