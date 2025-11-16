"""Supervised fine-tuning training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.metrics import accuracy, expected_correctness
from src.models import build_student_model
from src.training import BaseExperiment
from src.utils import (
    log_metrics,
    move_to_device,
    save_experiment_results,
    save_model,
    setup_logging,
)


class SupervisedExperiment(BaseExperiment):
    """Experiment for supervised fine-tuning."""
    
    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        """
        Initialize supervised experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.model.device)
        
        # Setup logging
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        
        # Create output directory
        self.output_dir = Path(config.training.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataloaders
        self.logger.info("Loading dataset and creating dataloaders...")
        self.dataloaders = create_dataloaders(
            dataset_cfg=config.dataset,
            model_name=config.model.student_model_name,
            training_cfg=config.training,
            unlabeled_pool_size=1000,  # Not used for supervised, but required by function
        )
        
        # Build student model
        self.logger.info(f"Building student model: {config.model.student_model_name}")
        self.model = build_student_model(config.model)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Setup scheduler (linear warmup + decay)
        warmup_steps = config.training.warmup_steps
        # Limit warmup steps to avoid too many scheduler steps
        max_warmup = len(self.dataloaders["train_labeled"]) * 2  # Max 2 batches worth
        warmup_steps = min(warmup_steps, max_warmup)
        
        if warmup_steps > 0:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Initialized experiment: {config.experiment_name}")
        self.logger.info(f"Train samples: {len(self.dataloaders['train_labeled'].dataset)}")
        if "dev" in self.dataloaders:
            self.logger.info(f"Dev samples: {len(self.dataloaders['dev'].dataset)}")
        if "test" in self.dataloaders:
            self.logger.info(f"Test samples: {len(self.dataloaders['test'].dataset)}")
        
        # Store training metrics
        self.training_metrics: list[dict[str, Any]] = []
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        self.model.train()
        
        train_loader = self.dataloaders["train_labeled"]
        dev_loader = self.dataloaders.get("dev")
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training loop
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = move_to_device(batch, self.device)
                
                # Extract batch info
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                # Forward pass
                logits = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                    labels=labels,
                )
                
                # Compute loss (cross-entropy)
                loss = F.cross_entropy(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step (only during warmup)
                max_warmup = len(train_loader) * 2  # Max 2 batches worth
                if self.scheduler is not None and batch_idx < min(self.config.training.warmup_steps, max_warmup):
                    self.scheduler.step()
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Evaluate on dev set
            if dev_loader is not None:
                dev_metrics = self._evaluate_on_loader(dev_loader, split="dev")
                epoch_metrics = {"loss": avg_loss, **dev_metrics}
                log_metrics(
                    step=f"train/epoch_{epoch + 1}",
                    metrics=epoch_metrics,
                    logger=self.logger,
                )
            else:
                epoch_metrics = {"loss": avg_loss}
                log_metrics(
                    step=f"train/epoch_{epoch + 1}",
                    metrics=epoch_metrics,
                    logger=self.logger,
                )
            
            # Store training metrics
            self.training_metrics.append({
                "epoch": epoch + 1,
                **epoch_metrics,
            })
        
        # Save final model
        self.logger.info("Saving final model...")
        save_model(
            model=self.model,
            output_path=self.output_dir,
            config=self.config,
        )
        self.logger.info(f"Model saved to {self.output_dir}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set (or validation if test has no labels).
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Prefer test set, but fall back to validation if test is not available
        # or if test set appears to have no ground truth labels
        eval_split = "test"
        if "test" not in self.dataloaders:
            if "dev" in self.dataloaders:
                self.logger.warning("No test set available, using validation set for evaluation")
                eval_split = "dev"
            else:
                self.logger.warning("No test or validation set available for evaluation")
                return {}
        elif "test" in self.dataloaders:
            # Check if test set has valid labels (not all zeros)
            # This is a heuristic - if all labels are 0, test set likely has no ground truth
            test_loader = self.dataloaders["test"]
            all_test_labels = []
            for batch in test_loader:
                labels = batch["labels"]
                all_test_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else [labels])
            
            if all_test_labels and all(label == 0 for label in all_test_labels):
                self.logger.warning(
                    "Test set appears to have no ground truth labels (all labels are 0). "
                    "Using validation set for evaluation instead."
                )
                if "dev" in self.dataloaders:
                    eval_split = "dev"
                else:
                    self.logger.warning("No validation set available, using test set anyway")
        
        self.logger.info(f"Evaluating on {eval_split} set...")
        eval_metrics = self._evaluate_on_loader(self.dataloaders[eval_split], split=eval_split)
        
        log_metrics(
            step=f"eval/{eval_split}",
            metrics=eval_metrics,
            logger=self.logger,
        )
        
        return eval_metrics
    
    def _evaluate_on_loader(
        self,
        loader: DataLoader,
        split: str = "eval",
    ) -> Dict[str, float]:
        """
        Evaluate model on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
            split: Split name for logging
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating on {split}"):
                # Move batch to device
                batch = move_to_device(batch, self.device)
                
                # Extract batch info
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                # Forward pass
                logits = self.model.forward(
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
        if len(all_predictions) == 0:
            self.logger.warning(f"No predictions collected for {split}")
            return {"accuracy": 0.0, "expected_correctness": 0.0}
        
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # Ensure same dtype for comparison
        all_predictions = all_predictions.astype(np.int64)
        all_labels = all_labels.astype(np.int64)
        
        # Debug info (only log once)
        if split == "test":
            self.logger.info(
                f"Test evaluation - predictions shape: {all_predictions.shape}, "
                f"labels shape: {all_labels.shape}, "
                f"predictions dtype: {all_predictions.dtype}, "
                f"labels dtype: {all_labels.dtype}, "
                f"unique predictions: {np.unique(all_predictions)}, "
                f"unique labels: {np.unique(all_labels)}, "
                f"first 10 predictions: {all_predictions[:10]}, "
                f"first 10 labels: {all_labels[:10]}"
            )
        
        # Compute metrics
        acc = accuracy(all_predictions, all_labels)
        exp_correct = expected_correctness(all_probs, all_labels)
        
        metrics = {
            "accuracy": acc,
            "expected_correctness": exp_correct,
        }
        
        self.model.train()  # Set back to train mode
        
        return metrics
