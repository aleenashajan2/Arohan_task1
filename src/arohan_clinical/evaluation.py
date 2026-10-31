"""Metric calculations used for held-out test evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None, "pr_auc": None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def confusion_counts(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])


def roc_points(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def pr_points(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return recall, precision


def metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results["models"].items():
        t = m["test_metrics"]
        rows.append({"Model": name, "Type": m.get("family", "baseline"), "Accuracy": t["accuracy"], "Precision": t["precision"],
                     "Recall (sensitivity)": t["recall"], "Specificity": t["specificity"], "F1": t["f1"],
                     "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"]})
    return pd.DataFrame(rows)
