"""Compare results from already trained models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import site
site.addsitedir(site.getusersitepackages())

from src.models.student import StudentMCQAModel
from src.utils import get_device, load_config
from scripts.test_model_examples import test_model_on_examples, create_test_examples
import torch


def load_and_test_model(checkpoint_dir: Path):
    """Load and test a model from checkpoint."""
    model_path = checkpoint_dir / "model.pt"
    config_path = checkpoint_dir / "config.json"
    
    if not model_path.exists() or not config_path.exists():
        return None
    
    try:
        # Load config
        experiment_config = load_config(config_path)
        model = StudentMCQAModel(experiment_config.model)
        
        # Load weights
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
        
        return {
            "model_name": experiment_config.model.student_model_name,
            "test_accuracy": test_results['accuracy'],
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Compare existing models."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     СРАВНЕНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    results_dir = Path("outputs/results")
    checkpoints_dir = Path("outputs/checkpoints")
    
    # Load JSON results
    print("📊 Загрузка результатов из JSON...")
    json_results = []
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            exp_name = json_file.stem
            final_metrics = data.get("final_metrics", {})
            config = data.get("config", {})
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            dataset_config = config.get("dataset", {})
            
            json_results.append({
                "experiment": exp_name,
                "model": model_config.get("student_model_name", "unknown"),
                "accuracy": final_metrics.get("accuracy", 0.0),
                "expected_correctness": final_metrics.get("expected_correctness", 0.0),
                "num_epochs": training_config.get("num_epochs", 0),
                "max_samples": dataset_config.get("max_samples", None),
            })
        except Exception as e:
            print(f"⚠ Ошибка загрузки {json_file}: {e}")
    
    # Test models on examples
    print("🧪 Тестирование моделей на примерах...")
    print()
    
    test_results = {}
    for checkpoint_dir in checkpoints_dir.iterdir():
        if checkpoint_dir.is_dir():
            # Skip nested dirs like student_v1
            if (checkpoint_dir / "model.pt").exists():
                print(f"  Тестирование: {checkpoint_dir.name}...")
                result = load_and_test_model(checkpoint_dir)
                if result:
                    test_results[checkpoint_dir.name] = result
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print()
    
    # Merge JSON and test results
    all_results = []
    for json_result in json_results:
        exp_name = json_result["experiment"]
        test_result = test_results.get(exp_name, {})
        
        all_results.append({
            **json_result,
            "test_accuracy": test_result.get("test_accuracy", 0.0),
        })
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # Print table
    print(f"{'Эксперимент':<30} {'Модель':<25} {'Accuracy':<12} {'Test Acc.':<12} {'Эпох':<8} {'Данных':<10}")
    print("-" * 110)
    
    for result in all_results:
        model_name = result["model"].split("/")[-1] if "/" in result["model"] else result["model"]
        max_samples = result["max_samples"] if result["max_samples"] else "all"
        
        print(
            f"{result['experiment']:<30} "
            f"{model_name[:24]:<25} "
            f"{result['accuracy']:.4f}      "
            f"{result['test_accuracy']:.4f}      "
            f"{result['num_epochs']:<8} "
            f"{str(max_samples):<10}"
        )
    
    print("-" * 110)
    print()
    
    # Best results
    if all_results:
        best_accuracy = max(all_results, key=lambda x: x["accuracy"])
        best_test = max(all_results, key=lambda x: x["test_accuracy"])
        
        print("🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
        print(f"  Лучшая accuracy: {best_accuracy['experiment']} ({best_accuracy['accuracy']:.4f})")
        print(f"  Лучшая test accuracy: {best_test['experiment']} ({best_test['test_accuracy']:.4f})")
        print()
    
    print(f"✅ Всего моделей: {len(all_results)}")
    
    return all_results


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

