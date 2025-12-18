"""Wrapper for teacher models and ensemble for multiple choice QA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from src.config import ModelConfig
from src.utils import get_device


class TeacherMCQAModel(nn.Module):
    """Wrapper for a single teacher model for multiple choice question answering."""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        device: str = "cpu",
    ) -> None:
        """
        Initialize teacher model.
        
        Args:
            model_name: Name of the teacher model
            num_labels: Number of labels/choices
            device: Device to use (default: "cpu")
        """
        super().__init__()
        self.model_name = model_name
        self.device = get_device(device)
        
        # Load model for multiple choice
        self.model = AutoModelForMultipleChoice.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)
        self.model.eval()  # Set to eval mode by default
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
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
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict probability distribution over choices with temperature.
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
            temperature: Temperature for softmax (default: 1.0)
        
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
            # Apply temperature scaling
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
        return probs


class TeacherEnsemble(nn.Module):
    """Ensemble of multiple teacher models for multiple choice QA."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        weights: list[float] | None = None,
    ) -> None:
        """
        Initialize teacher ensemble.
        
        Args:
            model_config: Model configuration
            weights: Ensemble weights for each teacher (if None, uniform weights)
        """
        super().__init__()
        self.model_config = model_config
        self.device = get_device(model_config.device)
        
        if not model_config.teacher_model_names:
            raise ValueError("At least one teacher model name is required")
        
        # Create teacher models
        self.teachers: list[TeacherMCQAModel] = []
        for teacher_name in model_config.teacher_model_names:
            teacher = TeacherMCQAModel(
                model_name=teacher_name,
                num_labels=model_config.num_labels,
                device=model_config.device,
            )
            self.teachers.append(teacher)
        
        # Set ensemble weights (uniform by default)
        num_teachers = len(self.teachers)
        if weights is None:
            self.weights = [1.0 / num_teachers] * num_teachers
        else:
            if len(weights) != num_teachers:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of teachers ({num_teachers})"
                )
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Convert to tensor for easier computation
        self.weights_tensor = torch.tensor(self.weights, device=self.device)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Returns averaged logits from all teachers.
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
        
        Returns:
            Averaged logits of shape (batch_size, num_options)
        """
        # Get logits from each teacher
        teacher_logits = []
        for teacher in self.teachers:
            logits = teacher.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            teacher_logits.append(logits)
        
        # Stack and average with weights
        # teacher_logits: list of (batch_size, num_options) tensors
        stacked_logits = torch.stack(teacher_logits, dim=0)  # (num_teachers, batch_size, num_options)
        
        # Apply weights and sum
        weighted_logits = stacked_logits * self.weights_tensor.view(-1, 1, 1)
        averaged_logits = weighted_logits.sum(dim=0)  # (batch_size, num_options)
        
        return averaged_logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict aggregated probability distribution over choices.
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
            temperature: Temperature for softmax (default: 1.0)
        
        Returns:
            Aggregated probabilities of shape (batch_size, num_options)
        """
        # Get probabilities from each teacher
        teacher_probs = []
        for teacher in self.teachers:
            probs = teacher.predict_proba(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
                temperature=temperature,
            )
            teacher_probs.append(probs)
        
        # Stack and average with weights
        stacked_probs = torch.stack(teacher_probs, dim=0)  # (num_teachers, batch_size, num_options)
        
        # Apply weights and sum
        weighted_probs = stacked_probs * self.weights_tensor.view(-1, 1, 1)
        averaged_probs = weighted_probs.sum(dim=0)  # (batch_size, num_options)
        
        return averaged_probs
    
    def get_per_teacher_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_options: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> list[torch.Tensor]:
        """
        Get logits from each teacher separately.
        
        Args:
            input_ids: Token indices of shape (batch_size * num_options, seq_len)
            attention_mask: Attention mask of shape (batch_size * num_options, seq_len)
            num_options: Number of options per example, shape (batch_size,)
            batch_size: Batch size (if None, inferred)
        
        Returns:
            List of logits from each teacher, each of shape (batch_size, num_options)
        """
        teacher_logits = []
        for teacher in self.teachers:
            logits = teacher.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            teacher_logits.append(logits)
        return teacher_logits
