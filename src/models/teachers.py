"""Обертка для teacher ensemble."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import KDConfig, ModelConfig


class TeacherEnsemble(nn.Module):
    """Ensemble из нескольких teacher моделей."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        kd_config: KDConfig,
    ) -> None:
        """
        Инициализирует teacher ensemble.
        
        Args:
            model_config: Конфигурация моделей
            kd_config: Конфигурация distillation
        """
        super().__init__()
        self.model_config = model_config
        self.kd_config = kd_config
        self.teachers: list[nn.Module] = []
        self.tokenizers: list[AutoTokenizer] = []
        
        # Загружаем все teacher модели
        for teacher_name in model_config.teacher_model_names:
            teacher = AutoModelForSequenceClassification.from_pretrained(
                teacher_name,
                num_labels=model_config.num_labels,
            )
            tokenizer = AutoTokenizer.from_pretrained(teacher_name)
            self.teachers.append(teacher)
            self.tokenizers.append(tokenizer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Прямой проход через ensemble.
        
        Args:
            input_ids: Индексы токенов
            attention_mask: Маска внимания
        
        Returns:
            Усредненные logits от всех teachers
        """
        # TODO: Реализовать forward pass через ensemble
        pass
    
    def predict_soft_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Генерирует soft labels с температурой.
        
        Args:
            input_ids: Индексы токенов
            attention_mask: Маска внимания
            temperature: Температура для softmax (если None, используется из config)
        
        Returns:
            Soft labels (softmax с температурой)
        """
        # TODO: Реализовать генерацию soft labels
        pass
    
    def compute_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет uncertainty для active learning.
        
        Args:
            input_ids: Индексы токенов
            attention_mask: Маска внимания
        
        Returns:
            Вектор uncertainty для каждого примера
        """
        # TODO: Реализовать вычисление uncertainty
        pass

