"""Обертка для student модели."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ModelConfig


class StudentModel(nn.Module):
    """Обертка для student модели из HuggingFace."""
    
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        """
        Инициализирует student модель.
        
        Args:
            config: Конфигурация модели
        """
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.student_model_name,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        
        if config.freeze_embeddings:
            # TODO: Заморозить embeddings
            pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Прямой проход модели.
        
        Args:
            input_ids: Индексы токенов
            attention_mask: Маска внимания
            labels: Метки для обучения (опционально)
        
        Returns:
            Словарь с outputs модели
        """
        # TODO: Реализовать forward pass
        pass
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Предсказания модели.
        
        Args:
            input_ids: Индексы токенов
            attention_mask: Маска внимания
        
        Returns:
            Logits предсказаний
        """
        # TODO: Реализовать предсказания
        pass

