"""CLI script for active learning loop (Expert-Loop v1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import default_medqa_experiment
from src.training.active_loop import ActiveLoopExperiment
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Active learning loop experiment (v1)"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="active_loop_v1",
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
    parser.add_argument(
        "--teacher_model_names",
        type=str,
        nargs="+",
        default=None,
        help="Teacher model names (default: bert-base-uncased roberta-base)",
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
    
    # KD arguments (optional)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for distillation (default: 4.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha weight for KD loss (default: 0.7)",
    )
    
    # Active loop arguments (optional)
    parser.add_argument(
        "--unlabeled_pool_size",
        type=int,
        default=None,
        help="Size of unlabeled pool (default: 1000)",
    )
    parser.add_argument(
        "--top_k_uncertain",
        type=int,
        default=None,
        help="Number of most uncertain examples to select (default: 50)",
    )
    parser.add_argument(
        "--uncertainty_metric",
        type=str,
        choices=["entropy", "margin"],
        default=None,
        help="Uncertainty metric (default: entropy)",
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
    """Main function to run active learning loop."""
    args = parse_args()
    
    # Create default config
    config = default_medqa_experiment(
        experiment_name=args.experiment_name,
        experiment_type="active_loop",
    )
    
    # Override with CLI arguments if provided
    if args.dataset_name is not None:
        config.dataset.dataset_name = args.dataset_name
    
    if args.student_model_name is not None:
        config.model.student_model_name = args.student_model_name
    
    if args.teacher_model_names is not None:
        config.model.teacher_model_names = args.teacher_model_names
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    
    if args.temperature is not None:
        if config.kd is None:
            raise ValueError("KDConfig is required for active loop experiment")
        config.kd.temperature = args.temperature
    
    if args.alpha is not None:
        if config.kd is None:
            raise ValueError("KDConfig is required for active loop experiment")
        config.kd.alpha = args.alpha
    
    if args.unlabeled_pool_size is not None:
        if config.active is None:
            raise ValueError("ActiveLoopConfig is required for active loop experiment")
        config.active.unlabeled_pool_size = args.unlabeled_pool_size
    
    if args.top_k_uncertain is not None:
        if config.active is None:
            raise ValueError("ActiveLoopConfig is required for active loop experiment")
        config.active.top_k_uncertain = args.top_k_uncertain
    
    if args.uncertainty_metric is not None:
        if config.active is None:
            raise ValueError("ActiveLoopConfig is required for active loop experiment")
        config.active.uncertainty_metric = args.uncertainty_metric
    
    config.training.seed = args.seed
    config.dataset.seed = args.seed
    
    # Validate config
    config.validate()
    
    # Set seed
    set_seed(config.training.seed)
    
    # Create experiment
    experiment = ActiveLoopExperiment(config)
    
    # Train
    experiment.train()
    
    # Evaluate
    final_metrics = experiment.evaluate()
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("Final Evaluation Metrics (Active Loop v1):")
    print("=" * 60)
    for metric_name, metric_value in final_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 60)
    print(f"Active Loop Configuration:")
    print(f"  Unlabeled pool size: {config.active.unlabeled_pool_size}")
    print(f"  Top-K uncertain: {config.active.top_k_uncertain}")
    print(f"  Uncertainty metric: {config.active.uncertainty_metric}")
    print(f"  KD Temperature: {config.kd.temperature}")
    print(f"  KD Alpha: {config.kd.alpha}")
    print(f"  Teacher models: {config.model.teacher_model_names}")
    print("=" * 60)


if __name__ == "__main__":
    main()
