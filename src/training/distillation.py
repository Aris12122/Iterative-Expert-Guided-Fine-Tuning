"""Knowledge distillation training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.metrics import accuracy, expected_correctness
from src.models import build_student_model, build_teacher_ensemble
from src.training import BaseExperiment
from src.utils import get_device, log_metrics, move_to_device, save_model, setup_logging


class KDExperiment(BaseExperiment):
    """Experiment for knowledge distillation from teacher ensemble."""
    
    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        """
        Initialize knowledge distillation experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = get_device(config.model.device)
        
        if config.kd is None:
            raise ValueError("KDConfig is required for KDExperiment")
        self.kd_config = config.kd
        
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
            unlabeled_pool_size=1000,  # Not used for KD, but required by function
        )
        
        # Build student model
        self.logger.info(f"Building student model: {config.model.student_model_name}")
        self.student_model = build_student_model(config.model)
        self.student_model.to(self.device)
        
        # Build teacher ensemble
        self.logger.info(f"Building teacher ensemble: {config.model.teacher_model_names}")
        self.teacher_ensemble = build_teacher_ensemble(config.model)
        self.teacher_ensemble.to(self.device)
        self.teacher_ensemble.eval()  # Teacher ensemble is always in eval mode
        
        # Setup optimizer (only for student)
        self.optimizer = AdamW(
            self.student_model.parameters(),
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
        
        self.logger.info(f"Initialized KD experiment: {config.experiment_name}")
        self.logger.info(f"Temperature: {self.kd_config.temperature}, Alpha: {self.kd_config.alpha}")
        self.logger.info(f"Train samples: {len(self.dataloaders['train_labeled'].dataset)}")
        if "dev" in self.dataloaders:
            self.logger.info(f"Dev samples: {len(self.dataloaders['dev'].dataset)}")
        if "test" in self.dataloaders:
            self.logger.info(f"Test samples: {len(self.dataloaders['test'].dataset)}")
        
        # Store training metrics
        self.training_metrics: list[dict[str, Any]] = []
    
    def train(self) -> None:
        """Train the student model through knowledge distillation."""
        self.logger.info("Starting knowledge distillation training...")
        self.student_model.train()
        self.teacher_ensemble.eval()  # Ensure teacher is in eval mode
        
        train_loader = self.dataloaders["train_labeled"]
        dev_loader = self.dataloaders.get("dev")
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training loop
            total_loss = 0.0
            total_ce_loss = 0.0
            total_kd_loss = 0.0
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
                
                # Student forward pass
                student_logits = self.student_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_logits = self.teacher_ensemble.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_options=num_options,
                        batch_size=batch_size,
                    )
                
                # Compute distillation loss
                loss, ce_loss, kd_loss = self._compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step (only during warmup)
                max_warmup = len(train_loader) * 2  # Max 2 batches worth
                if self.scheduler is not None and batch_idx < min(self.config.training.warmup_steps, max_warmup):
                    self.scheduler.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_kd_loss += kd_loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "ce": ce_loss.item(),
                    "kd": kd_loss.item(),
                })
            
            # Average losses
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0.0
            avg_kd_loss = total_kd_loss / num_batches if num_batches > 0 else 0.0
            
            # Evaluate on dev set
            if dev_loader is not None:
                dev_metrics = self._evaluate_on_loader(dev_loader, split="dev")
                epoch_metrics = {
                    "loss": avg_loss,
                    "ce_loss": avg_ce_loss,
                    "kd_loss": avg_kd_loss,
                    **dev_metrics,
                }
                log_metrics(
                    step=f"train/epoch_{epoch + 1}",
                    metrics=epoch_metrics,
                    logger=self.logger,
                )
            else:
                epoch_metrics = {
                    "loss": avg_loss,
                    "ce_loss": avg_ce_loss,
                    "kd_loss": avg_kd_loss,
                }
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
            model=self.student_model,
            output_path=self.output_dir,
            config=self.config,
        )
        self.logger.info(f"Model saved to {self.output_dir}")
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Loss = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(teacher_probs_T || student_probs_T)
        
        Args:
            student_logits: Student logits, shape (batch_size, num_options)
            teacher_logits: Teacher ensemble logits, shape (batch_size, num_options)
            labels: True labels, shape (batch_size,)
        
        Returns:
            Tuple of (total_loss, ce_loss, kd_loss)
        """
        temperature = self.kd_config.temperature
        alpha = self.kd_config.alpha
        
        # Hard loss: cross-entropy with true labels
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss: KL divergence between teacher and student distributions at temperature T
        # Apply temperature scaling
        student_logits_T = student_logits / temperature
        teacher_logits_T = teacher_logits / temperature
        
        # Compute softmax probabilities
        student_probs_T = F.log_softmax(student_logits_T, dim=-1)
        teacher_probs_T = F.softmax(teacher_logits_T, dim=-1)
        
        # KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
        # = sum(teacher * log(teacher)) - sum(teacher * log(student))
        # = -H(teacher) - sum(teacher * log(student))
        # Since we're minimizing, we compute: sum(teacher * log(student))
        kd_loss = F.kl_div(
            student_probs_T,
            teacher_probs_T,
            reduction="batchmean",
        )
        
        # Scale by T^2 as in the original KD paper
        kd_loss = kd_loss * (temperature ** 2)
        
        # Combine losses
        total_loss = (1 - alpha) * ce_loss + alpha * kd_loss
        
        return total_loss, ce_loss, kd_loss
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if "test" not in self.dataloaders:
            self.logger.warning("No test set available for evaluation")
            return {}
        
        self.logger.info("Evaluating on test set...")
        test_metrics = self._evaluate_on_loader(self.dataloaders["test"], split="test")
        
        log_metrics(
            step="eval/test",
            metrics=test_metrics,
            logger=self.logger,
        )
        
        return test_metrics
    
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
        self.student_model.eval()
        
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
                
                # Forward pass (student only for evaluation)
                logits = self.student_model.forward(
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
        
        metrics = {
            "accuracy": acc,
            "expected_correctness": exp_correct,
        }
        
        self.student_model.train()  # Set back to train mode
        
        return metrics
