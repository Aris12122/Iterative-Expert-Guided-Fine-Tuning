"""Configuration classes for all experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    dataset_name: str
    train_split: str = "train"
    val_split: Optional[str] = None
    test_split: Optional[str] = "test"
    max_seq_length: int = 512
    text_column: str = "text"
    label_column: str = "label"
    seed: int = 42
    max_samples: Optional[int] = None  # Limit dataset size for quick testing (None = use all)


@dataclass
class ModelConfig:
    """Model configuration (student and teacher)."""
    
    student_model_name: str
    teacher_model_names: list[str] = field(default_factory=list)
    num_labels: int = 2
    device: str = "cpu"


@dataclass
class TrainingConfig:
    """General training parameters."""
    
    batch_size: int = 8  # Reduced for CPU and quick testing
    num_epochs: int = 1  # Reduced for quick testing
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 10  # Reduced for quick testing
    max_grad_norm: float = 1.0
    eval_steps: int = 50  # Reduced for quick testing
    save_steps: int = 100
    logging_steps: int = 10  # Reduced for quick testing
    output_dir: str = "outputs/checkpoints"
    seed: int = 42
    num_threads: int = 2  # Limit CPU threads to avoid overloading


@dataclass
class KDConfig:
    """Knowledge distillation configuration."""
    
    alpha: float = 0.7
    temperature: float = 4.0


@dataclass
class ActiveLoopConfig:
    """Active learning loop configuration."""
    
    unlabeled_pool_size: int = 1000
    top_k_uncertain: int = 50
    uncertainty_metric: Literal["entropy", "margin"] = "entropy"
    max_active_iterations: int = 1


@dataclass
class ExperimentConfig:
    """Unified experiment configuration."""
    
    experiment_name: str
    experiment_type: Literal["supervised", "kd", "active_loop"]
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    kd: Optional[KDConfig] = None
    active: Optional[ActiveLoopConfig] = None
    
    def validate(self) -> None:
        """Validate configuration for compatibility."""
        if self.experiment_type not in ["supervised", "kd", "active_loop"]:
            raise ValueError(
                f"experiment_type must be 'supervised', 'kd' or 'active_loop', "
                f"got: {self.experiment_type}"
            )
        
        if self.experiment_type in ["kd", "active_loop"]:
            if self.kd is None:
                raise ValueError(
                    f"For experiment_type '{self.experiment_type}' KDConfig is required"
                )
            if not self.model.teacher_model_names:
                raise ValueError(
                    f"For experiment_type '{self.experiment_type}' "
                    "at least one teacher model is required"
                )
        
        if self.experiment_type == "active_loop":
            if self.active is None:
                raise ValueError(
                    "For experiment_type 'active_loop' ActiveLoopConfig is required"
                )
        
        if self.model.device != "cpu":
            raise ValueError(
                f"Project supports only CPU, got device: {self.model.device}"
            )


def default_medqa_experiment(
    experiment_name: str = "medqa_default",
    experiment_type: Literal["supervised", "kd", "active_loop"] = "supervised",
) -> ExperimentConfig:
    """
    Create an experiment configuration with reasonable defaults for MedQA/MedMCQA.
    
    Args:
        experiment_name: Experiment name
        experiment_type: Experiment type (supervised, kd, active_loop)
    
    Returns:
        ExperimentConfig with settings for medical QA tasks
    """
    dataset = DatasetConfig(
        dataset_name="medmcqa",
        train_split="train",
        val_split="validation",
        test_split="test",
        max_seq_length=512,
        text_column="question",
        label_column="correct",
        seed=42,
        max_samples=2500, # Limit to 1000 samples for quick testing
    )
    
    model = ModelConfig(
        student_model_name="distilbert-base-uncased",
        teacher_model_names=["bert-base-uncased", "roberta-base"] if experiment_type != "supervised" else [],
        num_labels=4,  # MedMCQA has 4 answer options
        device="cpu",
    )
    
    training = TrainingConfig(
        batch_size=8,  # Reduced for CPU and quick testing
        num_epochs=1,  # Reduced for quick testing
        learning_rate=2e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        warmup_steps=10,  # Reduced for quick testing
        max_grad_norm=1.0,
        eval_steps=50,  # Reduced for quick testing
        save_steps=100,
        logging_steps=10,  # Reduced for quick testing
        output_dir="outputs/checkpoints",
        seed=42,
        num_threads=2,  # Limit CPU threads
    )
    
    kd: Optional[KDConfig] = None
    if experiment_type in ["kd", "active_loop"]:
        kd = KDConfig(
            alpha=0.7,
            temperature=4.0,
        )
    
    active: Optional[ActiveLoopConfig] = None
    if experiment_type == "active_loop":
        active = ActiveLoopConfig(
            unlabeled_pool_size=100,  # Reduced for quick testing
            top_k_uncertain=20,  # Reduced for quick testing
            uncertainty_metric="entropy",
            max_active_iterations=1,
        )
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        experiment_type=experiment_type,
        dataset=dataset,
        model=model,
        training=training,
        kd=kd,
        active=active,
    )
    
    config.validate()
    return config
