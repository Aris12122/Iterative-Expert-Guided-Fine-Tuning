"""Metrics for model evaluation."""

from __future__ import annotations

import numpy as np
import torch


def accuracy(
    pred_indices: np.ndarray,
    true_indices: np.ndarray,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        pred_indices: Predicted label indices, shape (N,)
        true_indices: True label indices, shape (N,)
    
    Returns:
        Accuracy score between 0 and 1
    """
    if len(pred_indices) != len(true_indices):
        raise ValueError(
            f"Length mismatch: pred_indices ({len(pred_indices)}) "
            f"vs true_indices ({len(true_indices)})"
        )
    
    correct = np.sum(pred_indices == true_indices)
    total = len(pred_indices)
    
    return float(correct / total) if total > 0 else 0.0


def expected_correctness(
    probs: np.ndarray,
    true_indices: np.ndarray,
) -> float:
    """
    Compute expected correctness (average probability of correct answer).
    
    For each sample, takes the probability assigned to the correct answer,
    then averages over all samples.
    
    Args:
        probs: Probability distributions, shape (N, num_choices)
        true_indices: True label indices, shape (N,)
    
    Returns:
        Expected correctness score between 0 and 1
    """
    if probs.shape[0] != len(true_indices):
        raise ValueError(
            f"Length mismatch: probs ({probs.shape[0]}) "
            f"vs true_indices ({len(true_indices)})"
        )
    
    # Get probability of correct answer for each sample
    correct_probs = probs[np.arange(len(true_indices)), true_indices]
    
    # Average over all samples
    return float(np.mean(correct_probs))


def entropy(
    probs: np.ndarray,
) -> np.ndarray:
    """
    Compute entropy for each sample.
    
    Entropy = -sum(p * log(p)) over choices for each sample.
    
    Args:
        probs: Probability distributions, shape (N, num_choices)
        Must be valid probability distributions (sum to 1)
    
    Returns:
        Entropy values, shape (N,)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs_safe = probs + eps
    
    # Compute entropy: -sum(p * log(p))
    entropy_values = -np.sum(probs * np.log(probs_safe), axis=1)
    
    return entropy_values


def max_prob(
    probs: np.ndarray,
) -> np.ndarray:
    """
    Compute maximum probability for each sample.
    
    Useful for uncertainty estimation: uncertainty = 1 - max_prob.
    
    Args:
        probs: Probability distributions, shape (N, num_choices)
    
    Returns:
        Maximum probability values, shape (N,)
    """
    return np.max(probs, axis=1)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute average KL divergence between two probability distributions over a dataset.
    
    KL(p || q) = sum(p * log(p / q)) for each sample, then averaged.
    
    Args:
        p: First probability distributions, shape (N, num_choices)
        q: Second probability distributions, shape (N, num_choices)
    
    Returns:
        Average KL divergence
    """
    if p.shape != q.shape:
        raise ValueError(
            f"Shape mismatch: p {p.shape} vs q {q.shape}"
        )
    
    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-10
    p_safe = p + eps
    q_safe = q + eps
    
    # Normalize to ensure they sum to 1
    p_safe = p_safe / p_safe.sum(axis=1, keepdims=True)
    q_safe = q_safe / q_safe.sum(axis=1, keepdims=True)
    
    # Compute KL divergence: sum(p * log(p / q)) for each sample
    kl_per_sample = np.sum(p_safe * np.log(p_safe / q_safe), axis=1)
    
    # Average over all samples
    return float(np.mean(kl_per_sample))


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted label indices, shape (N,)
        labels: True label indices, shape (N,)
        probs: Probability distributions, shape (N, num_choices) (optional)
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    metrics = {
        "accuracy": accuracy(pred_np, labels_np),
    }
    
    # Add probability-based metrics if available
    if probs is not None:
        probs_np = probs.detach().cpu().numpy()
        metrics["expected_correctness"] = expected_correctness(probs_np, labels_np)
        
        # Compute average entropy and max_prob
        entropy_values = entropy(probs_np)
        max_prob_values = max_prob(probs_np)
        metrics["mean_entropy"] = float(np.mean(entropy_values))
        metrics["mean_max_prob"] = float(np.mean(max_prob_values))
        metrics["mean_uncertainty"] = float(np.mean(1.0 - max_prob_values))
    
    return metrics
