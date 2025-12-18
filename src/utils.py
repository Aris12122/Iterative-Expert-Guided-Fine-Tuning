"""Common utilities and helper functions."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import ExperimentConfig


def get_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    If device is None or "auto", automatically selects:
    - "cuda" if CUDA is available
    - "cpu" otherwise
    
    Args:
        device: Device string ("cpu", "cuda", "auto", or None)
    
    Returns:
        torch.device object
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def set_seed(seed: int, num_threads: int = 2) -> None:
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.
    
    Also limits CPU threads to avoid overloading the system.
    
    Args:
        seed: Seed value
        num_threads: Number of CPU threads to use (default: 2 to avoid CPU overload)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Limit CPU threads to avoid overloading the system
    torch.set_num_threads(num_threads)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    log_dir: str,
    experiment_name: str,
) -> logging.Logger:
    """
    Setup logging for an experiment.
    
    Args:
        log_dir: Directory for logs
        experiment_name: Experiment name
    
    Returns:
        Configured logger
    """
    log_path = Path(log_dir) / f"{experiment_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_metrics(
    step: str,
    metrics: dict[str, float],
    logger: logging.Logger | None = None,
    log_file: Path | None = None,
) -> dict[str, float]:
    """
    Log metrics to console and optionally to file.
    
    Args:
        step: Step identifier (e.g., "train/epoch_1", "eval/step_100")
        metrics: Dictionary of metric names to values
        logger: Logger instance (optional)
        log_file: Path to JSON log file (optional)
    
    Returns:
        The metrics dictionary (unchanged)
    """
    # Format metrics string
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    log_message = f"{step} - {metrics_str}"
    
    # Log to console/logger
    if logger is not None:
        logger.info(log_message)
    else:
        print(log_message)
    
    # Log to JSON file if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing logs or create new
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log entry
        log_entry = {
            "step": step,
            **metrics,
        }
        logs.append(log_entry)
        
        # Save
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    
    return metrics


def move_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Move all tensors in a batch to the specified device.
    
    Args:
        batch: Dictionary of tensors
        device: Target device
    
    Returns:
        Dictionary with tensors moved to device
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def save_config(
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        config: Experiment configuration
        output_path: Path for saving
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict
    config_dict = {
        "experiment_name": config.experiment_name,
        "experiment_type": config.experiment_type,
        "dataset": {
            "dataset_name": config.dataset.dataset_name,
            "train_split": config.dataset.train_split,
            "val_split": config.dataset.val_split,
            "test_split": config.dataset.test_split,
            "max_seq_length": config.dataset.max_seq_length,
            "text_column": config.dataset.text_column,
            "label_column": config.dataset.label_column,
            "seed": config.dataset.seed,
        },
        "model": {
            "student_model_name": config.model.student_model_name,
            "teacher_model_names": config.model.teacher_model_names,
            "num_labels": config.model.num_labels,
            "device": config.model.device,
        },
        "training": {
            "batch_size": config.training.batch_size,
            "num_epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "warmup_steps": config.training.warmup_steps,
            "max_grad_norm": config.training.max_grad_norm,
            "eval_steps": config.training.eval_steps,
            "save_steps": config.training.save_steps,
            "logging_steps": config.training.logging_steps,
            "output_dir": config.training.output_dir,
            "seed": config.training.seed,
            "num_threads": config.training.num_threads,
        },
    }
    
    if config.kd is not None:
        config_dict["kd"] = {
            "alpha": config.kd.alpha,
            "temperature": config.kd.temperature,
        }
    
    if config.active is not None:
        config_dict["active"] = {
            "unlabeled_pool_size": config.active.unlabeled_pool_size,
            "top_k_uncertain": config.active.top_k_uncertain,
            "uncertainty_metric": config.active.uncertainty_metric,
            "max_active_iterations": config.active.max_active_iterations,
        }
    
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config(
    config_path: Path,
) -> ExperimentConfig:
    """
    Load experiment configuration from JSON.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Loaded configuration
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Reconstruct config objects
    from src.config import (
        ActiveLoopConfig,
        DatasetConfig,
        KDConfig,
        ModelConfig,
        TrainingConfig,
    )
    
    dataset = DatasetConfig(**config_dict["dataset"])
    model = ModelConfig(**config_dict["model"])
    
    # Handle backward compatibility for num_threads
    training_dict = config_dict["training"].copy()
    if "num_threads" not in training_dict:
        training_dict["num_threads"] = 2  # Default value
    
    training = TrainingConfig(**training_dict)
    
    kd = None
    if "kd" in config_dict:
        kd = KDConfig(**config_dict["kd"])
    
    active = None
    if "active" in config_dict:
        active = ActiveLoopConfig(**config_dict["active"])
    
    config = ExperimentConfig(
        experiment_name=config_dict["experiment_name"],
        experiment_type=config_dict["experiment_type"],
        dataset=dataset,
        model=model,
        training=training,
        kd=kd,
        active=active,
    )
    
    return config


def save_model(
    model: torch.nn.Module,
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    """
    Save model and configuration.
    
    Args:
        model: Model to save
        output_path: Path for saving
        config: Experiment configuration
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = output_path / "config.json"
    save_config(config, config_path)


def load_model(
    model_path: Path,
    config: ExperimentConfig,
) -> torch.nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to checkpoint directory (should contain model.pt)
        config: Experiment configuration
    
    Returns:
        Loaded model
    """
    from src.models import build_student_model
    
    # Create model
    model = build_student_model(config.model)
    
    # Load state dict
    state_dict_path = model_path / "model.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Model file not found: {state_dict_path}")
    
    device = get_device(config.model.device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model


def save_experiment_results(
    experiment_name: str,
    config: ExperimentConfig,
    final_metrics: dict[str, float],
    training_metrics: list[dict[str, Any]] | None = None,
    results_dir: str | Path = "outputs/results",
) -> Path:
    """
    Save experiment results to a JSON file.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        final_metrics: Final evaluation metrics (e.g., accuracy, expected_correctness)
        training_metrics: List of training metrics per epoch (optional)
        results_dir: Directory to save results
    
    Returns:
        Path to saved results file
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        "experiment_name": experiment_name,
        "experiment_type": config.experiment_type,
        "config": {
            "dataset": {
                "dataset_name": config.dataset.dataset_name,
                "max_seq_length": config.dataset.max_seq_length,
                "max_samples": config.dataset.max_samples,
            },
            "model": {
                "student_model_name": config.model.student_model_name,
                "teacher_model_names": config.model.teacher_model_names,
                "num_labels": config.model.num_labels,
                "device": config.model.device,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "num_epochs": config.training.num_epochs,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "seed": config.training.seed,
            },
        },
        "final_metrics": final_metrics,
    }
    
    # Add KD config if present
    if config.kd is not None:
        results["config"]["kd"] = {
            "alpha": config.kd.alpha,
            "temperature": config.kd.temperature,
        }
    
    # Add active learning config if present
    if config.active is not None:
        results["config"]["active"] = {
            "unlabeled_pool_size": config.active.unlabeled_pool_size,
            "top_k_uncertain": config.active.top_k_uncertain,
            "uncertainty_metric": config.active.uncertainty_metric,
            "max_active_iterations": config.active.max_active_iterations,
        }
    
    # Add training metrics if provided
    if training_metrics is not None:
        results["training_metrics"] = training_metrics
    
    # Save to JSON
    results_file = results_path / f"{experiment_name}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results_file
