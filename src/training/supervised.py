"""Цикл supervised fine-tuning."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.metrics import compute_classification_metrics
from src.models.student import StudentModel
from src.utils import setup_logging


class SupervisedTrainer:
    """Тренер для supervised fine-tuning."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: StudentModel,
    ) -> None:
        """
        Инициализирует тренер.
        
        Args:
            config: Конфигурация эксперимента
            model: Student модель для обучения
        """
        self.config = config
        self.model = model
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        self.device = torch.device(config.device)
        self.model.to(self.device)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """
        Обучает модель.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации (опционально)
        
        Returns:
            Словарь с историей метрик
        """
        # TODO: Реализовать цикл обучения
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Обучает модель на одной эпохе.
        
        Args:
            train_loader: DataLoader для обучения
        
        Returns:
            Словарь с метриками эпохи
        """
        # TODO: Реализовать обучение на одной эпохе
        pass
    
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Оценивает модель на validation set.
        
        Args:
            val_loader: DataLoader для валидации
        
        Returns:
            Словарь с метриками
        """
        # TODO: Реализовать оценку модели
        pass

