"""CLI скрипт для active learning loop (Expert-Loop v1)."""

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
from src.models.student import StudentModel
from src.models.teachers import TeacherEnsemble
from src.training.active_loop import ActiveLoopTrainer
from src.utils import save_config, set_seed


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Active learning loop experiment (v1)"
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
    
    # Active loop arguments
    parser.add_argument("--initial_pool_size", type=int, default=100)
    parser.add_argument("--query_size", type=int, default=50)
    parser.add_argument("--query_strategy", type=str, default="uncertainty")
    parser.add_argument("--max_iterations", type=int, default=1)
    
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
    
    active_loop_config = ActiveLoopConfig(
        initial_pool_size=args.initial_pool_size,
        query_size=args.query_size,
        query_strategy=args.query_strategy,
        max_iterations=args.max_iterations,
    )
    
    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        experiment_type="active_loop",
        dataset=dataset_config,
        model=model_config,
        training=training_config,
        kd=kd_config,
        active_loop=active_loop_config,
    )
    
    experiment_config.validate()
    return experiment_config


def main() -> None:
    """Главная функция для запуска active learning loop."""
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
    trainer = ActiveLoopTrainer(config, student_model, teacher_ensemble)
    
    # Запускаем active learning цикл
    # TODO: Реализовать active learning цикл
    print(f"Запуск active learning loop для эксперимента: {config.experiment_name}")


if __name__ == "__main__":
    main()

