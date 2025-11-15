"""CLI скрипт для оценки сохраненных моделей."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.data import create_dataloader, load_dataset_from_hub
from src.metrics import compute_classification_metrics
from src.models.student import StudentModel
from src.utils import load_config, load_model, set_seed


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model"
    )
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--test_split", type=str, default="test")
    
    return parser.parse_args()


def main() -> None:
    """Главная функция для оценки модели."""
    args = parse_args()
    
    # Загружаем конфигурацию
    config = load_config(Path(args.config_path))
    
    # Устанавливаем seed
    set_seed(config.training.seed)
    
    # Загружаем модель
    model = load_model(Path(args.model_path), config)
    model.eval()
    
    # Загружаем тестовый датасет
    # TODO: Реализовать загрузку и обработку тестового датасета
    
    # Оцениваем модель
    # TODO: Реализовать оценку модели
    print(f"Оценка модели из: {args.model_path}")


if __name__ == "__main__":
    main()

