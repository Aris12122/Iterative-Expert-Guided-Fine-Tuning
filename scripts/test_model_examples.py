"""Script to test trained model on specific examples with readable output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig
from src.utils import load_config
from src.data import (
    MCQAExample,
    MultipleChoiceCollator,
    load_and_normalize_dataset,
    normalize_medmcqa_example,
)
from src.models.student import StudentMCQAModel
from src.utils import get_device, set_seed


def create_test_examples() -> list[MCQAExample]:
    """
    Create a set of test examples for manual inspection.
    
    Returns:
        List of test MCQA examples
    """
    return [
        MCQAExample(
            question="What is the most common cause of acute appendicitis?",
            options=[
                "Bacterial infection",
                "Viral infection",
                "Obstruction of the appendiceal lumen",
                "Dietary factors",
            ],
            correct_index=2,  # Obstruction is the most common cause
        ),
        MCQAExample(
            question="Which medication is first-line treatment for hypertension?",
            options=[
                "Aspirin",
                "ACE inhibitors",
                "Antibiotics",
                "Insulin",
            ],
            correct_index=1,  # ACE inhibitors
        ),
        MCQAExample(
            question="What is the normal range for blood pressure?",
            options=[
                "90/60 - 120/80 mmHg",
                "140/90 - 160/100 mmHg",
                "180/120 - 200/130 mmHg",
                "50/30 - 70/50 mmHg",
            ],
            correct_index=0,  # Normal is 90/60 - 120/80
        ),
        MCQAExample(
            question="What is the primary function of insulin?",
            options=[
                "Increase blood glucose levels",
                "Decrease blood glucose levels",
                "Regulate blood pressure",
                "Fight infections",
            ],
            correct_index=1,  # Decrease blood glucose
        ),
        MCQAExample(
            question="Which organ produces bile?",
            options=[
                "Stomach",
                "Liver",
                "Pancreas",
                "Kidney",
            ],
            correct_index=1,  # Liver
        ),
        MCQAExample(
            question="What is the normal heart rate for adults at rest?",
            options=[
                "30-50 beats per minute",
                "60-100 beats per minute",
                "120-150 beats per minute",
                "180-200 beats per minute",
            ],
            correct_index=1,  # 60-100 bpm
        ),
    ]


def format_example_output(
    example: MCQAExample,
    predicted_index: int,
    probabilities: list[float],
    correct: bool,
    example_num: int,
) -> str:
    """
    Format a single example for readable output.
    
    Args:
        example: The MCQA example
        predicted_index: Index of predicted answer
        probabilities: List of probabilities for each option
        correct: Whether prediction is correct
        example_num: Example number
    
    Returns:
        Formatted string
    """
    option_labels = ["A", "B", "C", "D"]
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"Пример {example_num}")
    lines.append("=" * 80)
    lines.append(f"\n❓ ВОПРОС:")
    lines.append(f"   {example.question}")
    lines.append(f"\n📋 ВАРИАНТЫ ОТВЕТОВ:")
    
    for i, (label, option, prob) in enumerate(
        zip(option_labels, example.options, probabilities)
    ):
        marker = "✅" if i == predicted_index else "  "
        correct_marker = "✓" if i == example.correct_index else " "
        lines.append(
            f"   {marker} {label}) {option} {correct_marker}"
            f" [Вероятность: {prob:.2%}]"
        )
    
    lines.append(f"\n🎯 ПРЕДСКАЗАНИЕ МОДЕЛИ: {option_labels[predicted_index]}")
    lines.append(f"   Текст: {example.options[predicted_index]}")
    lines.append(f"   Уверенность: {probabilities[predicted_index]:.2%}")
    
    lines.append(f"\n✅ ПРАВИЛЬНЫЙ ОТВЕТ: {option_labels[example.correct_index]}")
    lines.append(f"   Текст: {example.options[example.correct_index]}")
    
    if correct:
        lines.append(f"\n🎉 РЕЗУЛЬТАТ: ПРАВИЛЬНО!")
    else:
        lines.append(f"\n❌ РЕЗУЛЬТАТ: НЕПРАВИЛЬНО")
        lines.append(
            f"   Модель выбрала '{example.options[predicted_index]}' "
            f"вместо '{example.options[example.correct_index]}'"
        )
    
    lines.append("")
    return "\n".join(lines)


def test_model_on_examples(
    model: StudentMCQAModel,
    examples: list[MCQAExample],
    device: torch.device,
    max_examples: int | None = None,
) -> dict[str, any]:
    """
    Test model on a list of examples.
    
    Args:
        model: Trained model
        examples: List of examples to test
        device: Device to run on
        max_examples: Maximum number of examples to test (None = all)
    
    Returns:
        Dictionary with test results
    """
    model.eval()
    collator = MultipleChoiceCollator(
        tokenizer=model.tokenizer,
        max_length=512,
    )
    
    if max_examples is not None:
        examples = examples[:max_examples]
    
    correct = 0
    total = len(examples)
    results = []
    
    print(f"\n🧪 Тестирование модели на {total} примерах...\n")
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(examples, desc="Тестирование")):
            # Tokenize example
            batch_data = collator([example])
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            num_options = batch_data["num_options"].to(device)
            batch_size = batch_data["batch_size"]
            
            # Get model prediction
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            predicted_index = int(torch.argmax(probs, dim=-1).item())
            probabilities = probs[0].cpu().tolist()
            
            # Check if correct
            is_correct = predicted_index == example.correct_index
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                "example_num": idx + 1,
                "question": example.question,
                "options": example.options,
                "correct_index": example.correct_index,
                "predicted_index": predicted_index,
                "probabilities": probabilities,
                "correct": is_correct,
            })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def print_summary(test_results: dict[str, any]) -> None:
    """Print summary of test results."""
    print("\n" + "=" * 80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего примеров: {test_results['total']}")
    print(f"Правильных ответов: {test_results['correct']}")
    print(f"Точность: {test_results['accuracy']:.2%}")
    print("=" * 80)


def print_detailed_results(test_results: dict[str, any]) -> None:
    """Print detailed results for each example."""
    print("\n" + "=" * 80)
    print("📝 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    for result in test_results["results"]:
        example = MCQAExample(
            question=result["question"],
            options=result["options"],
            correct_index=result["correct_index"],
        )
        output = format_example_output(
            example=example,
            predicted_index=result["predicted_index"],
            probabilities=result["probabilities"],
            correct=result["correct"],
            example_num=result["example_num"],
        )
        print(output)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> tuple[StudentMCQAModel, ModelConfig]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file (if None, looks for config.json in same dir)
    
    Returns:
        Tuple of (model, config)
    """
    if config_path is None:
        config_path = checkpoint_path.parent / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    from src.config import ExperimentConfig
    experiment_config: ExperimentConfig = load_config(config_path)
    model_config = experiment_config.model
    
    # Create model
    model = StudentMCQAModel(model_config)
    
    # Load weights
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=get_device("cpu"))
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model, model_config


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test trained model on specific examples"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file (default: config.json in same dir as model)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medmcqa",
        help="Dataset name to load examples from (default: medmcqa)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use (default: validation)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to test from dataset (default: 10)",
    )
    parser.add_argument(
        "--use_test_examples",
        action="store_true",
        help="Use predefined test examples instead of dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load model
    model_path = Path(args.model_path)
    config_path = Path(args.config_path) if args.config_path else None
    
    print("📦 Загрузка модели...")
    try:
        model, model_config = load_model_from_checkpoint(model_path, config_path)
        print(f"✓ Модель загружена: {model_path}")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    device = get_device(model_config.device)
    model.to(device)
    print(f"✓ Устройство: {device}")
    
    # Load examples
    if args.use_test_examples:
        print("\n📝 Использование предопределенных тестовых примеров...")
        examples = create_test_examples()
    else:
        print(f"\n📚 Загрузка примеров из датасета {args.dataset_name}...")
        try:
            all_examples = load_and_normalize_dataset(
                dataset_name=args.dataset_name,
                split=args.split,
            )
            examples = all_examples[: args.num_examples]
            print(f"✓ Загружено {len(examples)} примеров")
        except Exception as e:
            print(f"⚠ Ошибка загрузки датасета: {e}")
            print("📝 Использование предопределенных тестовых примеров...")
            examples = create_test_examples()
    
    # Test model
    test_results = test_model_on_examples(
        model=model,
        examples=examples,
        device=device,
    )
    
    # Print results
    print_detailed_results(test_results)
    print_summary(test_results)
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()

