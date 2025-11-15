"""Конфигурации для всех экспериментов проекта."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    """Конфигурация датасета."""
    
    dataset_name: str
    task_type: str = "classification"  # classification, regression, etc.
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    val_split: Optional[str] = None  # Если None, будет создан из train
    test_split: Optional[str] = None
    val_size: float = 0.1  # Доля для validation, если val_split не указан
    max_length: int = 512
    seed: int = 42


@dataclass
class ModelConfig:
    """Конфигурация моделей (student и teacher)."""
    
    student_model_name: str
    teacher_model_names: list[str] = field(default_factory=list)
    num_labels: int = 2
    dropout: float = 0.1
    freeze_embeddings: bool = False


@dataclass
class TrainingConfig:
    """Общие параметры обучения."""
    
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "linear"  # linear, cosine, constant
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "outputs/checkpoints"
    seed: int = 42
    logging_steps: int = 100


@dataclass
class KDConfig:
    """Конфигурация knowledge distillation."""
    
    temperature: float = 4.0
    alpha: float = 0.7  # Вес для distillation loss (1-alpha для hard labels)
    use_hard_labels: bool = True
    ensemble_method: str = "mean"  # mean, voting, weighted_mean


@dataclass
class ActiveLoopConfig:
    """Конфигурация active learning цикла."""
    
    initial_pool_size: int = 100
    query_size: int = 50
    query_strategy: str = "uncertainty"  # uncertainty, entropy, margin
    use_teacher_uncertainty: bool = True
    max_iterations: int = 1  # Для v1: одна итерация
    min_confidence_threshold: float = 0.5


@dataclass
class ExperimentConfig:
    """Объединенная конфигурация эксперимента."""
    
    experiment_name: str
    experiment_type: str  # supervised, kd, active_loop
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    kd: Optional[KDConfig] = None
    active_loop: Optional[ActiveLoopConfig] = None
    device: str = "cpu"
    num_workers: int = 0  # Для CPU обычно 0
    pin_memory: bool = False
    
    def validate(self) -> None:
        """Валидирует конфигурацию на совместимость."""
        if self.experiment_type not in ["supervised", "kd", "active_loop"]:
            raise ValueError(
                f"experiment_type должен быть 'supervised', 'kd' или 'active_loop', "
                f"получено: {self.experiment_type}"
            )
        
        if self.experiment_type in ["kd", "active_loop"]:
            if self.kd is None:
                raise ValueError(
                    f"Для experiment_type '{self.experiment_type}' требуется KDConfig"
                )
            if not self.model.teacher_model_names:
                raise ValueError(
                    f"Для experiment_type '{self.experiment_type}' требуется "
                    "хотя бы одна teacher модель"
                )
        
        if self.experiment_type == "active_loop":
            if self.active_loop is None:
                raise ValueError(
                    "Для experiment_type 'active_loop' требуется ActiveLoopConfig"
                )
        
        if self.device != "cpu":
            raise ValueError(
                f"Проект поддерживает только CPU, получено device: {self.device}"
            )

