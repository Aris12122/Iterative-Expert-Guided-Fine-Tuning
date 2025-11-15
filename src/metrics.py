"""Metrics for model evaluation."""

from __future__ import annotations

from typing import Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = "macro",
) -> dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (logits or classes)
        labels: True labels
        average: Averaging method for F1, precision, recall
    
    Returns:
        Dictionary with metrics
    """
    # TODO: Implement metric computation
    pass


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute accuracy.
    
    Args:
        predictions: Model predictions
        labels: True labels
    
    Returns:
        Accuracy score
    """
    # TODO: Implement accuracy computation
    pass


def compute_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = "macro",
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Model predictions
        labels: True labels
        average: Averaging method
    
    Returns:
        F1 score
    """
    # TODO: Implement F1 score computation
    pass
