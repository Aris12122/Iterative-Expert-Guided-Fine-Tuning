"""CLI скрипт для knowledge distillation (Baseline 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    DatasetConfig,
    ExperimentConfig,
    KDConfig,
    ModelConfig,
    TrainingConfig,
)
from src.models.student import StudentModel
from src.models.teachers import TeacherEnsemble
from src.training.distillation import DistillationTrainer
from src.utils import save_config, set_seed


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation baseline experiment"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--max_length", type=int, default=512)
    
    # Model arguments
    parser.add_argument("--student_model_name", type=str, required=True)
    parser.add_argument("--teacher_model_names", type=str, nargs="+", required=True)
    parser.add_argument("--num_labels", type=int, default=2)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--experiment_name", type=str, required=True)
    
    # KD arguments
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--ensemble_method", type=str, default="mean")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def create_config(args: argparse.Namespace) -> ExperimentConfig:
    """Создает конфигурацию эксперимента из аргументов."""
    dataset_config = DatasetConfig(
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        seed=args.seed,
    )
    
    model_config = ModelConfig(
        student_model_name=args.student_model_name,
        teacher_model_names=args.teacher_model_names,
        num_labels=args.num_labels,
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    kd_config = KDConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        ensemble_method=args.ensemble_method,
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type="kd",
        dataset=dataset_config,
        model=model_config,
        training=training_config,
        kd=kd_config,
        active_loop=None,
    )
    
    experiment_config.validate()
    return experiment_config


def main() -> None:
    """Главная функция для запуска knowledge distillation."""
    args = parse_args()
    config = create_config(args)
    
    # Устанавливаем seed
    set_seed(config.training.seed)
    
    # Создаем директорию для вывода
    output_dir = Path(config.training.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфигурацию
    save_config(config, output_dir / "config.json")
    
    # Загружаем датасет
    # TODO: Реализовать загрузку и обработку датасета
    
    # Создаем модели
    student_model = StudentModel(config.model)
    teacher_ensemble = TeacherEnsemble(config.model, config.kd)
    
    # Создаем тренер
    trainer = DistillationTrainer(config, student_model, teacher_ensemble)
    
    # Обучаем модель
    # TODO: Реализовать обучение
    print(f"Запуск knowledge distillation для эксперимента: {config.experiment_name}")


if __name__ == "__main__":
    main()

