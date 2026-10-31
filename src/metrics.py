"""Métriques au niveau patch, centrées sur les faux négatifs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.95) -> float:
    """Plus grand seuil qui garantit une sensibilité >= target (choisi sur la validation)."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def patch_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "auc_roc": float(roc_auc_score(y, p)) if both else np.nan,
        "auc_pr": float(average_precision_score(y, p)) if both else np.nan,
        "threshold": thr,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "fn": int(fn),
        "fp": int(fp),
    }


def report(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Métriques par split et par centre. `df` doit contenir split, center, label, prob."""
    rows = []
    for (split, center), g in df.groupby(["split", "center"]):
        rows.append({"split": split, "center": center, **patch_metrics(g.label.values, g.prob.values, thr)})
    for split, g in df.groupby("split"):
        rows.append({"split": split, "center": "all", **patch_metrics(g.label.values, g.prob.values, thr)})
    return pd.DataFrame(rows).sort_values(["split", "center"], key=lambda s: s.astype(str)).reset_index(drop=True)
