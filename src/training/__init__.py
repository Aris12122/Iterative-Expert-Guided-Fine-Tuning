"""Модули обучения."""

from src.training.active_loop import ActiveLoopTrainer
from src.training.distillation import DistillationTrainer
from src.training.supervised import SupervisedTrainer

__all__ = ["SupervisedTrainer", "DistillationTrainer", "ActiveLoopTrainer"]

