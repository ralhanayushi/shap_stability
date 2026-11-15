"""Evaluation module for churn prediction models.

Addresses SRS Requirements:
- REQ-F-28: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Log Loss computation.
- REQ-F-29: Strict test set isolation (evaluated once after tuning).
- REQ-F-30: Model comparison performance table.
- REQ-F-31: Validation-based threshold tuning (maximizing F1).
- REQ-F-32: Customer-level prediction persistence (probabilities and binary flags).
- REQ-F-33: Confusion matrix reporting and serialization.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find classification threshold on validation set that maximizes F1 score (REQ-F-31)."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return round(best_thresh, 3)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics (REQ-F-28)."""
    y_pred = (y_prob >= threshold).astype(int)

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = roc_auc_score(y_true, y_prob) if has_both_classes else float("nan")
    pr_auc = average_precision_score(y_true, y_prob) if has_both_classes else float("nan")

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "threshold": round(float(threshold), 3),
    }


def evaluate_models(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Evaluate models on validation and test sets, select thresholds, and export results."""
    records = []
    predictions_df = pd.DataFrame(index=X_test.index)
    predictions_df["actual_is_churn"] = y_test.values
    confusion_matrices = {}

    optimal_thresholds = {}

    for name, model in models.items():
        # Validation evaluation and threshold selection (REQ-F-31)
        val_probs = model.predict_proba(X_val)[:, 1]
        best_t = find_optimal_threshold(y_val.values, val_probs)
        optimal_thresholds[name] = best_t

        val_metrics_default = compute_metrics(y_val.values, val_probs, threshold=0.5)
        records.append({"model": name, "split": "validation (t=0.5)", **val_metrics_default})

        val_metrics_tuned = compute_metrics(y_val.values, val_probs, threshold=best_t)
        records.append({"model": name, "split": f"validation (t={best_t})", **val_metrics_tuned})

        # Test evaluation strictly once with tuned threshold (REQ-F-29)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= best_t).astype(int)

        test_metrics = compute_metrics(y_test.values, test_probs, threshold=best_t)
        records.append({"model": name, "split": "test", **test_metrics})

        # Save individual predictions (REQ-F-32)
        predictions_df[f"{name}_prob"] = test_probs
        predictions_df[f"{name}_pred"] = test_pred

        # Confusion Matrix (REQ-F-33)
        cm = confusion_matrix(y_test.values, test_pred)
        confusion_matrices[name] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

    # Prediction agreement analysis across models
    model_names = list(models.keys())
    if len(model_names) == 3:
        m1, m2, m3 = model_names
        p1, p2, p3 = predictions_df[f"{m1}_pred"], predictions_df[f"{m2}_pred"], predictions_df[f"{m3}_pred"]
        all_agree = (p1 == p2) & (p2 == p3)
        predictions_df["all_models_agree"] = all_agree
        logger.info(f"Model prediction agreement across all 3 models: {all_agree.mean() * 100:.2f}%")

    metrics_df = pd.DataFrame(records)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
        predictions_df.to_csv(output_dir / "test_predictions.csv")
        with open(output_dir / "confusion_matrices.json", "w") as f:
            json.dump(confusion_matrices, f, indent=2)
        with open(output_dir / "optimal_thresholds.json", "w") as f:
            json.dump(optimal_thresholds, f, indent=2)

    return metrics_df, predictions_df, confusion_matrices
