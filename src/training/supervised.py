"""Supervised fine-tuning training loop."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.metrics import compute_classification_metrics
from src.models.student import StudentModel
from src.utils import setup_logging


class SupervisedTrainer:
    """Trainer for supervised fine-tuning."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: StudentModel,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            config: Experiment configuration
            model: Student model for training
        """
        self.config = config
        self.model = model
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        self.device = torch.device(config.model.device)
        self.model.to(self.device)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation (optional)
        
        Returns:
            Dictionary with metrics history
        """
        # TODO: Implement training loop
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
