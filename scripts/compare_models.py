"""Script to train and compare different model architectures."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add user site-packages to path
import site
site.addsitedir(site.getusersitepackages())

from src.config import default_medqa_experiment
from src.training.supervised import SupervisedExperiment
from src.utils import set_seed, save_experiment_results

# Import test script functions
import importlib.util
test_module_path = project_root / "scripts" / "test_model_examples.py"
spec = importlib.util.spec_from_file_location("test_model_examples", test_module_path)
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
test_model_on_examples = test_module.test_model_on_examples
create_test_examples = test_module.create_test_examples

from src.models.student import StudentMCQAModel
from src.utils import get_device
import torch


# Список моделей для сравнения (только легкие модели для быстрого теста)
MODELS_TO_TEST = [
    "distilbert-base-uncased",  # Маленькая и быстрая
    # "bert-base-uncased",      # Слишком большая для быстрого теста
    # "roberta-base",           # Слишком большая для быстрого теста
]


def create_config_for_model(model_name: str, max_samples: int = 200, num_epochs: int = 3):
    """Create config for specific model."""
    config = default_medqa_experiment(
        experiment_name=f"compare_{model_name.replace('/', '_')}",
        experiment_type="supervised",
    )
    
    config.model.student_model_name = model_name
    config.dataset.max_samples = max_samples
    config.training.batch_size = 8  # Уменьшен для экономии памяти
    config.training.num_epochs = num_epochs
    config.training.warmup_steps = 10
    config.training.eval_steps = 15
    config.training.logging_steps = 5
    config.training.num_threads = 4
    config.model.device = "cpu"
    
    return config


def train_and_evaluate_model(model_name: str, max_samples: int = 200, num_epochs: int = 3):
    """Train a model and return results."""
    print("=" * 80)
    print(f"🤖 ОБУЧЕНИЕ МОДЕЛИ: {model_name}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Create config
    config = create_config_for_model(model_name, max_samples, num_epochs)
    config.validate()
    
    print(f"📋 Конфигурация:")
    print(f"  - Модель: {model_name}")
    print(f"  - Данных: {config.dataset.max_samples} примеров")
    print(f"  - Эпох: {config.training.num_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print()
    
    # Set seed
    set_seed(config.training.seed, num_threads=config.training.num_threads)
    
    # Train
    print("🏋️ Обучение...")
    train_start = time.time()
    experiment = SupervisedExperiment(config)
    experiment.train()
    train_time = time.time() - train_start
    print(f"✓ Обучение завершено за {train_time:.1f} секунд ({train_time/60:.1f} минут)")
    print()
    
    # Evaluate
    print("📊 Оценка...")
    eval_start = time.time()
    final_metrics = experiment.evaluate()
    eval_time = time.time() - eval_start
    print(f"✓ Оценка завершена за {eval_time:.1f} секунд")
    print()
    
    # Test on examples
    print("🧪 Тестирование на примерах...")
    test_start = time.time()
    
    model_path = Path(config.training.output_dir) / config.experiment_name / "model.pt"
    test_accuracy = 0.0
    
    if model_path.exists():
        try:
            from src.utils import load_config
            config_path = model_path.parent / "config.json"
            if config_path.exists():
                experiment_config = load_config(config_path)
                model = StudentMCQAModel(experiment_config.model)
                
                checkpoint = torch.load(model_path, map_location=get_device("cpu"))
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                device = get_device("cpu")
                model.to(device)
                model.eval()
                
                examples = create_test_examples()
                test_results = test_model_on_examples(
                    model=model,
                    examples=examples,
                    device=device,
                    max_examples=6,
                )
                test_accuracy = test_results['accuracy']
        except Exception as e:
            print(f"⚠ Ошибка при тестировании: {e}")
    
    test_time = time.time() - test_start
    total_time = time.time() - start_time
    
    # Save results
    save_experiment_results(
        experiment_name=config.experiment_name,
        config=config,
        final_metrics=final_metrics,
        training_metrics=experiment.training_metrics,
    )
    
    print(f"✓ Тестирование завершено за {test_time:.1f} секунд")
    print()
    
    return {
        "model_name": model_name,
        "train_time": train_time,
        "eval_time": eval_time,
        "test_time": test_time,
        "total_time": total_time,
        "accuracy": final_metrics.get("accuracy", 0.0),
        "expected_correctness": final_metrics.get("expected_correctness", 0.0),
        "test_accuracy": test_accuracy,
    }


def main():
    """Compare multiple models."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     СРАВНЕНИЕ РАЗНЫХ МОДЕЛЕЙ                                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    overall_start = time.time()
    results = []
    
    # Параметры для быстрого теста (уменьшены для экономии времени)
    max_samples = 150  # Немного уменьшено
    num_epochs = 2  # Уменьшено для быстрого сравнения
    
    print(f"📊 Параметры теста:")
    print(f"  - Данных: {max_samples} примеров")
    print(f"  - Эпох: {num_epochs}")
    print(f"  - Моделей для теста: {len(MODELS_TO_TEST)}")
    print()
    
    # Train each model
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'='*80}")
        print(f"МОДЕЛЬ {i}/{len(MODELS_TO_TEST)}")
        print(f"{'='*80}\n")
        
        try:
            result = train_and_evaluate_model(model_name, max_samples, num_epochs)
            results.append(result)
        except Exception as e:
            print(f"❌ Ошибка при обучении {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model_name": model_name,
                "error": str(e),
            })
    
    # Print comparison
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print()
    
    # Table header
    print(f"{'Модель':<30} {'Accuracy':<12} {'Exp. Corr.':<12} {'Test Acc.':<12} {'Время':<12}")
    print("-" * 80)
    
    # Table rows
    for result in results:
        if "error" in result:
            print(f"{result['model_name']:<30} {'ERROR':<12} {'-':<12} {'-':<12} {'-':<12}")
        else:
            print(
                f"{result['model_name']:<30} "
                f"{result['accuracy']:.4f}      "
                f"{result['expected_correctness']:.4f}      "
                f"{result['test_accuracy']:.4f}      "
                f"{result['total_time']:.1f}с"
            )
    
    print("-" * 80)
    print(f"{'ИТОГО':<30} {'-':<12} {'-':<12} {'-':<12} {overall_time:.1f}с")
    print()
    
    # Find best model
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best_accuracy = max(valid_results, key=lambda x: x['accuracy'])
        best_test = max(valid_results, key=lambda x: x['test_accuracy'])
        fastest = min(valid_results, key=lambda x: x['total_time'])
        
        print("🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"  Лучшая accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
        print(f"  Лучшая test accuracy: {best_test['model_name']} ({best_test['test_accuracy']:.4f})")
        print(f"  Самая быстрая: {fastest['model_name']} ({fastest['total_time']:.1f}с)")
        print()
    
    print(f"⏱️  Общее время: {overall_time:.1f} секунд ({overall_time/60:.1f} минут)")
    
    if overall_time > 600:
        print("⚠ ПРЕДУПРЕЖДЕНИЕ: Время превысило 10 минут!")
    else:
        print("✅ Уложились в лимит 10 минут!")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Сравнение завершено!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

