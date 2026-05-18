"""Metric helpers only; no plotting and no model training."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from ..config import WORDS
except ImportError:  # Allows importing from a notebook whose cwd is projet_fil_rouge/
    from config import WORDS


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def confusion(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)


def classification_summary(y_true, y_pred, target_names=WORDS):
    """Return numeric metrics as a plain dictionary."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    report_target_names = None
    if target_names is not None and len(target_names) == len(labels):
        report_target_names = list(target_names)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_target_names,
            zero_division=0,
            output_dict=True,
        ),
    }
