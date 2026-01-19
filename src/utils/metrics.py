"""
HARBench Evaluation Metrics

Metrics for evaluating Human Activity Recognition models.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.

    This is the primary evaluation metric for HARBench.
    Macro-averaging treats all classes equally regardless of their frequency.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        Macro F1 score (0.0 to 1.0)
    """
    return f1_score(y_true, y_pred, average='macro')


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        Accuracy (0.0 to 1.0)
    """
    return accuracy_score(y_true, y_pred)


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute F1 score for each class.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        F1 score for each class
    """
    return f1_score(y_true, y_pred, average=None)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)

    Returns:
        Confusion matrix (n_classes, n_classes)
    """
    return confusion_matrix(y_true, y_pred)
