"""Wrapper for student model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ModelConfig


class StudentModel(nn.Module):
    """Wrapper for student model from HuggingFace."""
    
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        """
        Initialize student model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.student_model_name,
            num_labels=config.num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            labels: Labels for training (optional)
        
        Returns:
            Dictionary with model outputs
        """
        # TODO: Implement forward pass
        pass
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Model predictions.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
        
        Returns:
            Prediction logits
        """
        # TODO: Implement predictions
        pass
