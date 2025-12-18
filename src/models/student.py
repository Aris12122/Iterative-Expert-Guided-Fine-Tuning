"""Wrapper for student model for multiple choice QA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from src.config import ModelConfig
from src.utils import get_device


class StudentMCQAModel(nn.Module):
    """Wrapper for student model for multiple choice question answering."""
    
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
        self.device = get_device(config.device)
        
        # Load model for multiple choice
        self.model = AutoModelForMultipleChoice.from_pretrained(
            config.student_model_name,
            num_labels=config.num_labels,
        )
        self.model.to(self.device)
        # Keep in train mode by default (will be set to eval in predict methods)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Reshapes flat batch from collator (batch_size * num_options, seq_len)
        into format expected by MultipleChoice model (batch_size, num_options, seq_len).
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred from num_options)
            labels: Labels for training, shape (batch_size,) (optional)
        
        Returns:
            Logits of shape (batch_size, num_options)
        """
        # Infer batch_size and num_options if not provided
        if num_options is not None:
            if batch_size is None:
                batch_size = num_options.shape[0]
            num_choices = int(num_options[0].item())
        else:
            # Try to infer from input shape
            flat_batch_size = input_ids.shape[0]
            if batch_size is None:
                # Assume 4 options by default (common for MCQA)
                num_choices = 4
                batch_size = flat_batch_size // num_choices
            else:
                num_choices = flat_batch_size // batch_size
        
        # Reshape from (batch_size * num_options, seq_len) to (batch_size, num_options, seq_len)
        seq_len = input_ids.shape[1]
        input_ids = input_ids.view(batch_size, num_choices, seq_len)
        attention_mask = attention_mask.view(batch_size, num_choices, seq_len)
        
        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass through MultipleChoice model
        if labels is not None:
            labels = labels.to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Extract logits (shape: batch_size, num_choices)
        logits = outputs.logits
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Predict probability distribution over choices.
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
        
        Returns:
            Probabilities of shape (batch_size, num_options)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict_label(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Predict label indices (argmax over choices).
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
        
        Returns:
            Predicted label indices of shape (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            predictions = torch.argmax(logits, dim=-1)
        return predictions
