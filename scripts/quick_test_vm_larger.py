"""Quick test script for VM with larger parameters - trains and tests in under 10 minutes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add user site-packages to path (for --user installs)
import site
site.addsitedir(site.getusersitepackages())

from src.config import default_medqa_experiment
from src.training.supervised import SupervisedExperiment
from src.utils import set_seed, save_experiment_results

# Import test script function
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


def larger_test_config():
    """Create config optimized for larger quick testing (< 10 minutes)."""
    config = default_medqa_experiment(
        experiment_name="quick_test_vm_larger",
        experiment_type="supervised",
    )
    
    # Larger settings for better results (still within 10 min limit)
    config.dataset.max_samples = 200  # Увеличено с 100 до 200
    config.training.batch_size = 16  # Оставляем разумный batch size
    config.training.num_epochs = 3  # Увеличено с 1 до 3 эпох
    config.training.warmup_steps = 10  # Немного больше warmup
    config.training.eval_steps = 15  # Оценка почаще
    config.training.logging_steps = 5  # Логирование
    config.training.num_threads = 4  # Используем больше потоков на VM
    
    # Use CPU (no GPU needed for quick test)
    config.model.device = "cpu"
    
    return config


def main():
    """Run larger quick test: train and evaluate in under 10 minutes."""
    start_time = time.time()
    print("=" * 80)
    print("🚀 БЫСТРЫЙ ТЕСТ НА VM С БОЛЬШИМИ ПАРАМЕТРАМИ (< 10 минут)")
    print("=" * 80)
    print()
    
    # Create config
    print("📋 Создание конфигурации...")
    config = larger_test_config()
    config.validate()
    print(f"✓ Конфигурация создана: {config.experiment_name}")
    print(f"  - Данных: {config.dataset.max_samples} примеров")
    print(f"  - Эпох: {config.training.num_epochs}")
    print(f"  - Batch size: {config.training.batch_size}")
    print()
    
    # Set seed
    set_seed(config.training.seed, num_threads=config.training.num_threads)
    
    # Train
    print("🏋️ Начало обучения...")
    train_start = time.time()
    experiment = SupervisedExperiment(config)
    experiment.train()
    train_time = time.time() - train_start
    print(f"✓ Обучение завершено за {train_time:.1f} секунд ({train_time/60:.1f} минут)")
    print()
    
    # Quick evaluation
    print("📊 Быстрая оценка на валидации...")
    eval_start = time.time()
    final_metrics = experiment.evaluate()
    eval_time = time.time() - eval_start
    print(f"✓ Оценка завершена за {eval_time:.1f} секунд")
    print()
    
    # Print metrics
    print("=" * 80)
    print("📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 80)
    for metric_name, metric_value in final_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("=" * 80)
    print()
    
    # Test on examples
    print("🧪 Тестирование на примерах...")
    test_start = time.time()
    
    # Get model
    model_path = Path(config.training.output_dir) / config.experiment_name / "model.pt"
    if not model_path.exists():
        print(f"⚠ Модель не найдена: {model_path}")
        test_results = {"total": 0, "correct": 0, "accuracy": 0.0}
    else:
        # Load model for testing
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
            
            # Test on examples
            examples = create_test_examples()
            test_results = test_model_on_examples(
                model=model,
                examples=examples,
                device=device,
                max_examples=6,
            )
        else:
            print("⚠ Config не найден, пропускаем тестирование на примерах")
            test_results = {"total": 0, "correct": 0, "accuracy": 0.0}
    
    test_time = time.time() - test_start
    
    print(f"✓ Тестирование завершено за {test_time:.1f} секунд")
    print()
    
    # Print test summary
    print("=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 80)
    print(f"Всего примеров: {test_results['total']}")
    print(f"Правильных: {test_results['correct']}")
    print(f"Точность: {test_results['accuracy']:.2%}")
    print("=" * 80)
    print()
    
    # Save results
    results_file = save_experiment_results(
        experiment_name=config.experiment_name,
        config=config,
        final_metrics=final_metrics,
        training_metrics=experiment.training_metrics,
    )
    print(f"💾 Результаты сохранены: {results_file}")
    print()
    
    # Total time
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"⏱️ ОБЩЕЕ ВРЕМЯ: {total_time:.1f} секунд ({total_time/60:.1f} минут)")
    print("=" * 80)
    
    if total_time > 600:  # 10 minutes
        print("⚠ ПРЕДУПРЕЖДЕНИЕ: Время превысило 10 минут!")
    else:
        print("✅ Уложились в лимит 10 минут!")
    
    return {
        "total_time": total_time,
        "train_time": train_time,
        "eval_time": eval_time,
        "test_time": test_time,
        "metrics": final_metrics,
        "test_accuracy": test_results['accuracy'],
    }


if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✅ Тест успешно завершен!")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

