"""Diagnostic script to check data loading and model training."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from datasets import load_dataset
from src.config import default_medqa_experiment
from src.data import load_splits, normalize_medmcqa_example


def check_dataset_labels() -> None:
    """Check if dataset has valid labels."""
    print("=" * 60)
    print("Проверка загрузки данных и меток")
    print("=" * 60)
    
    # Load raw dataset
    print("\n1. Загрузка сырого датасета MedMCQA...")
    try:
        dataset = load_dataset("medmcqa", "A", split="train")
    except ValueError:
        dataset = load_dataset("medmcqa", split="train")
    
    print(f"   Всего примеров: {len(dataset)}")
    
    # Check first 10 examples
    print("\n2. Проверка первых 10 примеров:")
    for i in range(min(10, len(dataset))):
        example = dataset[i]
        cop = example.get("cop")
        correct = example.get("correct")
        answer_idx = example.get("answer_idx")
        
        print(f"\n   Пример {i}:")
        print(f"     cop: {cop} (type: {type(cop)})")
        print(f"     correct: {correct} (type: {type(correct)})")
        print(f"     answer_idx: {answer_idx} (type: {type(answer_idx)})")
        
        # Normalize
        normalized = normalize_medmcqa_example(example)
        print(f"     Нормализованный correct_index: {normalized.correct_index}")
    
    # Check label distribution
    print("\n3. Распределение меток в сыром датасете:")
    cop_values = []
    for i in range(min(1000, len(dataset))):
        example = dataset[i]
        cop = example.get("cop")
        if cop is not None:
            cop_values.append(cop)
    
    if cop_values:
        unique, counts = np.unique(cop_values, return_counts=True)
        print(f"   Уникальные значения cop: {dict(zip(unique, counts))}")
        print(f"   Всего ненулевых значений: {len(cop_values)} из {min(1000, len(dataset))}")
    else:
        print("   ВНИМАНИЕ: Все значения cop равны None!")
    
    # Check normalized labels
    print("\n4. Распределение нормализованных меток:")
    normalized_examples = [normalize_medmcqa_example(dataset[i]) for i in range(min(1000, len(dataset)))]
    normalized_labels = [ex.correct_index for ex in normalized_examples]
    unique, counts = np.unique(normalized_labels, return_counts=True)
    print(f"   Уникальные значения: {dict(zip(unique, counts))}")
    
    # Check if all labels are 0
    if all(label == 0 for label in normalized_labels):
        print("\n   ⚠️  ПРОБЛЕМА: Все нормализованные метки равны 0!")
        print("   Это означает, что метки не загружаются правильно.")
    else:
        print(f"\n   ✓ Метки выглядят нормально (не все равны 0)")
    
    # Check using load_splits
    print("\n5. Проверка через load_splits:")
    config = default_medqa_experiment(experiment_name="test", experiment_type="supervised")
    config.dataset.max_samples = 100  # Limit for quick check
    
    splits = load_splits(config.dataset, unlabeled_pool_size=50)
    
    train_labels = [ex.correct_index for ex in splits["train_labeled"]]
    dev_labels = [ex.correct_index for ex in splits["dev"]] if splits["dev"] else []
    
    print(f"   Train examples: {len(splits['train_labeled'])}")
    print(f"   Dev examples: {len(splits['dev']) if splits['dev'] else 0}")
    
    if train_labels:
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"   Train labels distribution: {dict(zip(unique, counts))}")
    
    if dev_labels:
        unique, counts = np.unique(dev_labels, return_counts=True)
        print(f"   Dev labels distribution: {dict(zip(unique, counts))}")


def check_model_training() -> None:
    """Check if model training is working correctly."""
    print("\n" + "=" * 60)
    print("Проверка обучения модели")
    print("=" * 60)
    
    import torch
    from src.config import default_medqa_experiment
    from src.data import create_dataloaders
    from src.models import build_student_model
    
    config = default_medqa_experiment(
        experiment_name="diagnostic_test",
        experiment_type="supervised",
    )
    config.dataset.max_samples = 100  # Small dataset for quick test
    config.training.batch_size = 4
    config.training.num_epochs = 1
    
    print("\n1. Создание DataLoaders...")
    dataloaders = create_dataloaders(
        dataset_cfg=config.dataset,
        model_name=config.model.student_model_name,
        training_cfg=config.training,
        unlabeled_pool_size=50,
    )
    
    train_loader = dataloaders["train_labeled"]
    print(f"   Train batches: {len(train_loader)}")
    
    print("\n2. Проверка первого батча:")
    batch = next(iter(train_loader))
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   labels: {batch['labels']}")
    print(f"   labels dtype: {batch['labels'].dtype}")
    print(f"   num_options: {batch['num_options']}")
    print(f"   batch_size: {batch['batch_size']}")
    
    print("\n3. Создание модели...")
    model = build_student_model(config.model)
    device = torch.device("cpu")
    model.to(device)
    
    print("\n4. Тестовый forward pass:")
    batch = move_to_device(batch, device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    num_options = batch["num_options"]
    batch_size = batch["batch_size"]
    
    model.eval()
    with torch.no_grad():
        logits = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_options=num_options,
            batch_size=batch_size,
        )
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits: {logits}")
        print(f"   Predictions (argmax): {torch.argmax(logits, dim=-1)}")
        print(f"   True labels: {labels}")
        
        # Check if predictions match labels
        predictions = torch.argmax(logits, dim=-1)
        matches = (predictions == labels).sum().item()
        print(f"   Совпадений: {matches} из {len(labels)}")
    
    print("\n5. Тестовый шаг обучения:")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Forward pass
    logits = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_options=num_options,
        batch_size=batch_size,
        labels=labels,
    )
    
    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Check loss after one step
    model.eval()
    with torch.no_grad():
        logits_after = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_options=num_options,
            batch_size=batch_size,
        )
        loss_after = torch.nn.functional.cross_entropy(logits_after, labels)
        print(f"   Loss after one step: {loss_after.item():.4f}")
        
        if loss_after < loss:
            print("   ✓ Loss уменьшился - обучение работает!")
        else:
            print("   ⚠️  Loss не уменьшился - возможна проблема")


def move_to_device(batch, device):
    """Helper function to move batch to device."""
    import torch
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


if __name__ == "__main__":
    check_dataset_labels()
    check_model_training()
    
    print("\n" + "=" * 60)
    print("Диагностика завершена")
    print("=" * 60)

