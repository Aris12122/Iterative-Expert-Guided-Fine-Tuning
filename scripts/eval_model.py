"""CLI script for evaluating saved models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.metrics import accuracy, expected_correctness
from src.models import build_student_model
from src.utils import load_config, set_seed
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model"
    )
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    
    return parser.parse_args()


def main() -> None:
    """Main function to evaluate a model."""
    args = parse_args()
    
    # Load configuration
    config = load_config(Path(args.config_path))
    
    # Set seed
    set_seed(config.training.seed)
    
    # Load model
    model = build_student_model(config.model)
    model_path = Path(args.model_path)
    if (model_path / "model.pt").exists():
        state_dict = torch.load(model_path / "model.pt", map_location=config.model.device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path / 'model.pt'}")
    
    model.eval()
    device = torch.device(config.model.device)
    model.to(device)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        dataset_cfg=config.dataset,
        model_name=config.model.student_model_name,
        training_cfg=config.training,
        unlabeled_pool_size=1000,
    )
    
    # Evaluate on test set
    if "test" not in dataloaders:
        print("Warning: No test set available for evaluation")
        return
    
    test_loader = dataloaders["test"]
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            num_options = batch["num_options"]
            batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
            
            # Forward pass
            logits = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_options=num_options,
                batch_size=batch_size,
            )
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Compute metrics
    acc = accuracy(all_predictions, all_labels)
    exp_correct = expected_correctness(all_probs, all_labels)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Expected Correctness: {exp_correct:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

