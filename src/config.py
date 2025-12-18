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
    device: str = "auto"  # "auto", "cpu", or "cuda" - auto selects GPU if available


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
        max_samples=500, # Limit to 1000 samples for quick testing
    )
    
    model = ModelConfig(
        student_model_name="distilbert-base-uncased",
        teacher_model_names=["bert-base-uncased", "roberta-base"] if experiment_type != "supervised" else [],
        num_labels=4,  # MedMCQA has 4 answer options
        device="auto",  # Automatically use GPU if available
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


def improved_medqa_experiment(
    experiment_name: str = "medqa_improved",
    experiment_type: Literal["supervised", "kd", "active_loop"] = "supervised",
    use_full_dataset: bool = True,
    gpu_optimized: bool = True,
) -> ExperimentConfig:
    """
    Create an improved experiment configuration with better models for GPU training.
    
    Uses more powerful models (BERT-base/RoBERTa-base for student, BERT-large/RoBERTa-large
    for teachers) and GPU-optimized settings.
    
    Args:
        experiment_name: Experiment name
        experiment_type: Experiment type (supervised, kd, active_loop)
        use_full_dataset: If True, use full dataset (max_samples=None)
        gpu_optimized: If True, use GPU-optimized batch sizes and settings
    
    Returns:
        ExperimentConfig with improved settings for medical QA tasks
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
        max_samples=None if use_full_dataset else 500,
    )
    
    # Use more powerful models
    model = ModelConfig(
        # Better student model (BERT-base instead of DistilBERT)
        student_model_name="bert-base-uncased",  # or "roberta-base"
        # Large teacher models for better knowledge distillation
        teacher_model_names=[
            "bert-large-uncased",
            "roberta-large",
        ] if experiment_type != "supervised" else [],
        num_labels=4,  # MedMCQA has 4 answer options
        device="auto",  # Automatically use GPU if available
    )
    
    # GPU-optimized training settings
    if gpu_optimized:
        training = TrainingConfig(
            batch_size=32,  # Larger batch size for GPU
            num_epochs=5,  # More epochs for better convergence
            learning_rate=2e-5,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            warmup_steps=500,  # More warmup steps
            max_grad_norm=1.0,
            eval_steps=500,  # More frequent evaluation
            save_steps=1000,
            logging_steps=100,
            output_dir="outputs/checkpoints",
            seed=42,
            num_threads=4,  # More threads on GPU machine
        )
    else:
        # CPU-friendly settings
        training = TrainingConfig(
            batch_size=8,
            num_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            max_grad_norm=1.0,
            eval_steps=200,
            save_steps=500,
            logging_steps=50,
            output_dir="outputs/checkpoints",
            seed=42,
            num_threads=2,
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
            unlabeled_pool_size=1000 if use_full_dataset else 100,
            top_k_uncertain=50 if use_full_dataset else 20,
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
