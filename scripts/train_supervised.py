"""CLI script for supervised fine-tuning (Baseline 1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import default_medqa_experiment
from src.training.supervised import SupervisedExperiment
from src.utils import save_experiment_results, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning baseline experiment"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="supervised_baseline",
        help="Name of the experiment",
    )
    
    # Dataset arguments (optional, will use defaults if not provided)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name (default: medmcqa)",
    )
    
    # Model arguments (optional)
    parser.add_argument(
        "--student_model_name",
        type=str,
        default=None,
        help="Student model name (default: distilbert-base-uncased)",
    )
    
    # Training arguments (optional)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 3)",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function to run supervised training."""
    args = parse_args()
    
    # Create default config
    config = default_medqa_experiment(
        experiment_name=args.experiment_name,
        experiment_type="supervised",
    )
    
    # Override with CLI arguments if provided
    if args.dataset_name is not None:
        config.dataset.dataset_name = args.dataset_name
    
    if args.student_model_name is not None:
        config.model.student_model_name = args.student_model_name
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    
    config.training.seed = args.seed
    config.dataset.seed = args.seed
    
    # Validate config
    config.validate()
    
    # Set seed and limit CPU threads
    set_seed(config.training.seed, num_threads=config.training.num_threads)
    
    # Create experiment
    experiment = SupervisedExperiment(config)
    
    # Train
    experiment.train()
    
    # Evaluate
    final_metrics = experiment.evaluate()
    
    # Save results to JSON file
    results_file = save_experiment_results(
        experiment_name=config.experiment_name,
        config=config,
        final_metrics=final_metrics,
        training_metrics=experiment.training_metrics,
    )
    
    # Print final metrics
    print("\n" + "=" * 50)
    print("Final Evaluation Metrics:")
    print("=" * 50)
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 50)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
