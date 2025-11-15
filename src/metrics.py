"""Метрики для оценки моделей."""

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
    Вычисляет метрики классификации.
    
    Args:
        predictions: Предсказания модели (logits или классы)
        labels: Истинные метки
        average: Метод усреднения для F1, precision, recall
    
    Returns:
        Словарь с метриками
    """
    # TODO: Реализовать вычисление метрик
    pass


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Вычисляет accuracy.
    
    Args:
        predictions: Предсказания модели
        labels: Истинные метки
    
    Returns:
        Accuracy score
    """
    # TODO: Реализовать вычисление accuracy
    pass


def compute_f1_score(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    average: str = "macro",
) -> float:
    """
    Вычисляет F1 score.
    
    Args:
        predictions: Предсказания модели
        labels: Истинные метки
        average: Метод усреднения
    
    Returns:
        F1 score
    """
    # TODO: Реализовать вычисление F1 score
    pass

