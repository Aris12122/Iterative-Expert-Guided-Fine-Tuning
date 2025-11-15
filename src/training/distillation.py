"""Knowledge distillation training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.metrics import compute_classification_metrics
from src.models.student import StudentModel
from src.models.teachers import TeacherEnsemble
from src.utils import setup_logging


class DistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        student_model: StudentModel,
        teacher_ensemble: TeacherEnsemble,
    ) -> None:
        """
        Initialize trainer for distillation.
        
        Args:
            config: Experiment configuration
            student_model: Student model
            teacher_ensemble: Teacher ensemble
        """
        self.config = config
        self.student_model = student_model
        self.teacher_ensemble = teacher_ensemble
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        self.device = torch.device(config.model.device)
        self.student_model.to(self.device)
        self.teacher_ensemble.to(self.device)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """
        Train student model through distillation.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
        
        Returns:
            Dictionary with metrics history
        """
        # TODO: Implement distillation training loop
        pass
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for distillation.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher ensemble
            hard_labels: Hard labels (optional)
        
        Returns:
            Loss value
        """
        # TODO: Implement distillation loss computation
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training
        
        Returns:
            Dictionary with epoch metrics
        """
        # TODO: Implement one epoch training
        pass
    
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader: DataLoader for validation
        
        Returns:
            Dictionary with metrics
        """
        # TODO: Implement model evaluation
        pass
