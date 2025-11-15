"""Active learning loop with distillation."""

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
    """Trainer for active learning loop with distillation."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        student_model: StudentModel,
        teacher_ensemble: TeacherEnsemble,
    ) -> None:
        """
        Initialize trainer for active loop.
        
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
        
        if config.active is None:
            raise ValueError("ActiveLoopConfig is required for ActiveLoopTrainer")
        self.active_config = config.active
    
    def run_active_loop(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        """
        Run active learning loop.
        
        Args:
            train_dataset: Full training dataset
            val_loader: DataLoader for validation (optional)
        
        Returns:
            Dictionary with metrics history
        """
        # TODO: Implement active learning loop
        # 1. Create initial pool
        # 2. Train teacher ensemble on initial pool
        # 3. Compute uncertainty on remaining examples
        # 4. Select examples for query
        # 5. Train student through distillation on expanded pool
        pass
    
    def create_initial_pool(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> Subset:
        """
        Create initial pool of examples.
        
        Args:
            dataset: Full dataset
        
        Returns:
            Subset with initial pool
        """
        # TODO: Implement initial pool creation
        pass
    
    def query_examples(
        self,
        unlabeled_dataset: torch.utils.data.Dataset,
        query_size: int,
    ) -> list[int]:
        """
        Select examples for query based on uncertainty.
        
        Args:
            unlabeled_dataset: Dataset with unlabeled examples
            query_size: Number of examples to query
        
        Returns:
            List of indices of selected examples
        """
        # TODO: Implement example query strategy
        pass
    
    def compute_uncertainty_scores(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> torch.Tensor:
        """
        Compute uncertainty scores for all examples.
        
        Args:
            dataset: Dataset for evaluation
        
        Returns:
            Tensor with uncertainty scores
        """
        # TODO: Implement uncertainty computation
        pass
