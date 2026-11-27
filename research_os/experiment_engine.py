"""Part 6: Experiment Planning & Analysis Engine — real ROC/PR curves, stats, error analysis & run tracking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.schemas import ExperimentRecord, ResearchProject


def compute_roc_and_pr_curves(
    df: pd.DataFrame,
    y_true_col: str,
    y_score_col: str,
    positive_label: str = "1",
) -> dict[str, Any]:
    """
    Compute exact binary ROC curve (FPR, TPR, AUROC) and Precision-Recall curve (Recall, Precision, AUPRC)
    using pure NumPy/Python from actual uploaded scores (no SciPy DLL dependency).
    """
    sub = df[[y_true_col, y_score_col]].dropna()
    if sub.empty:
        raise ValueError("Selected ground-truth and score columns contain no valid rows.")

    y_true = [1 if str(v).strip() == str(positive_label).strip() else 0 for v in sub[y_true_col]]
    scores = pd.to_numeric(sub[y_score_col], errors="coerce")
    valid_mask = ~scores.isna()
    y_true = [yt for yt, ok in zip(y_true, valid_mask) if ok]
    y_score = [float(s) for s, ok in zip(scores, valid_mask) if ok]

    pos_total = sum(y_true)
    neg_total = len(y_true) - pos_total
    if pos_total == 0 or neg_total == 0:
        raise ValueError(f"ROC/PR require both positive ('{positive_label}') and negative samples.")

    # Sort descending by score
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    thresholds = sorted(set(y_score), reverse=True)

    fprs = [0.0]
    tprs = [0.0]
    precisions = [1.0]
    recalls = [0.0]

    tp = 0
    fp = 0
    idx = 0
    n = len(paired)
    for thr in thresholds:
        while idx < n and paired[idx][0] >= thr:
            if paired[idx][1] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        tpr = tp / pos_total
        fpr = fp / neg_total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        fprs.append(round(fpr, 4))
        tprs.append(round(tpr, 4))
        recalls.append(round(tpr, 4))
        precisions.append(round(prec, 4))

    fprs.append(1.0)
    tprs.append(1.0)

    # Trapezoidal AUC
    auroc = float(np.trapezoid(tprs, fprs)) if hasattr(np, "trapezoid") else float(np.trapz(tprs, fprs))
    auprc = float(np.trapezoid(precisions, recalls)) if hasattr(np, "trapezoid") else float(np.trapz(precisions, recalls))

    return {
        "samples": len(y_true),
        "positives": pos_total,
        "negatives": neg_total,
        "auroc": round(abs(auroc), 4),
        "auprc": round(abs(auprc), 4),
        "fpr": fprs,
        "tpr": tprs,
        "recall": recalls,
        "precision": precisions,
    }


def compute_statistical_summary(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    """Compute real sample size, mean, std, and 95% confidence interval per experimental group."""
    if group_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame()
    work = df[[group_col, metric_col]].copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return pd.DataFrame()

    rows = []
    for grp, grp_df in work.groupby(group_col):
        vals = grp_df[metric_col].tolist()
        n = len(vals)
        mean_v = sum(vals) / n
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / max(1, n - 1)) if n > 1 else 0.0
        ci95 = 1.96 * (std_v / math.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {
                "Group / Model": str(grp),
                "N": n,
                "Mean": round(mean_v, 4),
                "Std Dev": round(std_v, 4),
                "95% CI (+/-)": round(ci95, 4),
                "Min": round(min(vals), 4),
                "Max": round(max(vals), 4),
            }
        )
    return pd.DataFrame(rows)


def extract_error_analysis_rows(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    limit: int = 25,
) -> pd.DataFrame:
    """Return actual misclassified rows from uploaded predictions for error analysis."""
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        return pd.DataFrame()
    mask = df[y_true_col].astype(str) != df[y_pred_col].astype(str)
    return df[mask].head(limit)


def explain_actual_results(project: ResearchProject) -> str:
    """Generate a grounded summary strictly from actual uploaded experimental datasets."""
    er = project.experiment_results
    if not er.datasets and not er.computed_classification_metrics:
        return "No actual measured experimental datasets have been uploaded yet. Planned experiments remain unverified."

    lines: list[str] = []
    for ds in er.datasets:
        lines.append(f"- Measured Dataset '{ds.filename}' ({ds.row_count} rows, {len(ds.columns)} columns).")
        for col, stats in list(ds.numeric_summary.items())[:4]:
            lines.append(
                f"  - {col}: mean={stats.get('mean')}, std={stats.get('std')}, min={stats.get('min')}, max={stats.get('max')}"
            )
    if er.computed_classification_metrics:
        m = er.computed_classification_metrics
        lines.append(
            f"- Computed Classification Metrics (N={m.get('samples')}): "
            f"Accuracy={m.get('accuracy')}, Macro-Precision={m.get('precision_macro')}, "
            f"Macro-Recall={m.get('recall_macro')}, Macro-F1={m.get('f1_macro')}."
        )
    planned_cnt = sum(1 for r in project.experiment_records if r.status == "Planned")
    measured_cnt = sum(1 for r in project.experiment_records if r.status in ("Completed", "Actual Measured"))
    if project.experiment_records:
        lines.append(f"- Experiment Registry: {measured_cnt} Actual/Completed vs {planned_cnt} Planned.")
    return "\n".join(lines)


def sync_default_experiment_records(project: ResearchProject) -> list[ExperimentRecord]:
    """Populate initial planned/measured experiment status records from Stage 8 & Stage 9 without duplicating."""
    existing_names = {r.name for r in project.experiment_records}
    for bl in project.experiment_design.baselines:
        name = f"Baseline: {bl.get('name', 'Baseline')}"
        if name not in existing_names:
            project.experiment_records.append(
                ExperimentRecord(name=name, status="Planned", configuration=bl.get("type", ""), notes=bl.get("rationale", ""))
            )
            existing_names.add(name)
    for ds in project.experiment_results.datasets:
        name = f"Measured Dataset: {ds.filename}"
        if name not in existing_names:
            project.experiment_records.append(
                ExperimentRecord(name=name, status="Actual Measured", configuration=f"{ds.row_count} rows", notes=", ".join(ds.columns[:6]))
            )
            existing_names.add(name)
    return project.experiment_records
