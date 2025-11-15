"""CLI скрипт для supervised fine-tuning (Baseline 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    ActiveLoopConfig,
    DatasetConfig,
    ExperimentConfig,
    KDConfig,
    ModelConfig,
    TrainingConfig,
)
from src.data import create_dataloader, load_dataset_from_hub, split_dataset, tokenize_dataset
from src.models.student import StudentModel
from src.training.supervised import SupervisedTrainer
from src.utils import save_config, set_seed


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning baseline experiment"
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--max_length", type=int, default=512)
    
    # Model arguments
    parser.add_argument("--student_model_name", type=str, required=True)
    parser.add_argument("--num_labels", type=int, default=2)
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--experiment_name", type=str, required=True)
    
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
        num_labels=args.num_labels,
    )
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type="supervised",
        dataset=dataset_config,
        model=model_config,
        training=training_config,
        kd=None,
        active_loop=None,
    )
    
    experiment_config.validate()
    return experiment_config


def main() -> None:
    """Главная функция для запуска supervised обучения."""
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
    
    # Создаем модель
    student_model = StudentModel(config.model)
    
    # Создаем тренер
    trainer = SupervisedTrainer(config, student_model)
    
    # Обучаем модель
    # TODO: Реализовать обучение
    print(f"Запуск supervised обучения для эксперимента: {config.experiment_name}")


if __name__ == "__main__":
    main()

