"""Wrapper for teacher ensemble."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import KDConfig, ModelConfig


class TeacherEnsemble(nn.Module):
    """Ensemble of multiple teacher models."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        kd_config: KDConfig,
    ) -> None:
        """
        Initialize teacher ensemble.
        
        Args:
            model_config: Model configuration
            kd_config: Distillation configuration
        """
        super().__init__()
        self.model_config = model_config
        self.kd_config = kd_config
        self.teachers: list[nn.Module] = []
        self.tokenizers: list[AutoTokenizer] = []
        
        # Load all teacher models
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
        Forward pass through ensemble.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
        
        Returns:
            Averaged logits from all teachers
        """
        # TODO: Implement forward pass through ensemble
        pass
    
    def predict_soft_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Generate soft labels with temperature.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            temperature: Temperature for softmax (if None, use from config)
        
        Returns:
            Soft labels (softmax with temperature)
        """
        # TODO: Implement soft label generation
        pass
    
    def compute_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute uncertainty for active learning.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
        
        Returns:
            Uncertainty vector for each example
        """
        # TODO: Implement uncertainty computation
        pass
