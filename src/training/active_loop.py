"""Цикл active learning с distillation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from src.config import ExperimentConfig
from src.models.student import StudentModel
from src.models.teachers import TeacherEnsemble
from src.training.distillation import DistillationTrainer
from src.training.supervised import SupervisedTrainer
from src.utils import setup_logging


class ActiveLoopTrainer:
    """Тренер для active learning цикла с distillation."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        student_model: StudentModel,
        teacher_ensemble: TeacherEnsemble,
    ) -> None:
        """
        Инициализирует тренер для active loop.
        
        Args:
            config: Конфигурация эксперимента
            student_model: Student модель
            teacher_ensemble: Teacher ensemble
        """
        self.config = config
        self.student_model = student_model
        self.teacher_ensemble = teacher_ensemble
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        self.device = torch.device(config.device)
        
        if config.active_loop is None:
            raise ValueError("ActiveLoopConfig требуется для ActiveLoopTrainer")
        self.active_config = config.active_loop
    
    def run_active_loop(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """
        Запускает active learning цикл.
        
        Args:
            train_dataset: Полный тренировочный датасет
            val_loader: DataLoader для валидации (опционально)
        
        Returns:
            Словарь с историей метрик
        """
        # TODO: Реализовать active learning цикл
        # 1. Создать начальный пул
        # 2. Обучить teacher ensemble на начальном пуле
        # 3. Вычислить uncertainty на оставшихся примерах
        # 4. Выбрать примеры для запроса
        # 5. Обучить student через distillation на расширенном пуле
        pass
    
    def create_initial_pool(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> Subset:
        """
        Создает начальный пул примеров.
        
        Args:
            dataset: Полный датасет
        
        Returns:
            Subset с начальным пулом
        """
        # TODO: Реализовать создание начального пула
        pass
    
    def query_examples(
        self,
        unlabeled_dataset: torch.utils.data.Dataset,
        query_size: int,
    ) -> list[int]:
        """
        Выбирает примеры для запроса на основе uncertainty.
        
        Args:
            unlabeled_dataset: Датасет с непомеченными примерами
            query_size: Количество примеров для запроса
        
        Returns:
            Список индексов выбранных примеров
        """
        # TODO: Реализовать стратегию запроса примеров
        pass
    
    def compute_uncertainty_scores(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> torch.Tensor:
        """
        Вычисляет uncertainty scores для всех примеров.
        
        Args:
            dataset: Датасет для оценки
        
        Returns:
            Тензор с uncertainty scores
        """
        # TODO: Реализовать вычисление uncertainty
        pass

