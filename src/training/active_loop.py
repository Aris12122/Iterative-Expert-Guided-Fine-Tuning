"""Active learning loop with distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data import MCQAExample, MultipleChoiceDataset, create_dataloaders, load_splits
from src.metrics import accuracy, entropy, expected_correctness, max_prob
from src.models import build_student_model, build_teacher_ensemble
from src.training import BaseExperiment
from src.training.supervised import SupervisedExperiment
from src.utils import log_metrics, move_to_device, save_model, setup_logging


class CombinedDataset(Dataset):
    """Dataset combining hard-labeled and soft-labeled examples."""
    
    def __init__(
        self,
        hard_examples: list[MCQAExample],
        soft_examples: list[MCQAExample],
        soft_probs: list[torch.Tensor],
    ) -> None:
        """
        Initialize combined dataset.
        
        Args:
            hard_examples: Examples with hard labels
            soft_examples: Examples with soft labels from teacher
            soft_probs: Soft label probabilities for soft_examples
        """
        self.hard_examples = hard_examples
        self.soft_examples = soft_examples
        self.soft_probs = soft_probs
    
    def __len__(self) -> int:
        """Return total number of examples."""
        return len(self.hard_examples) + len(self.soft_examples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get item by index."""
        if idx < len(self.hard_examples):
            # Hard-labeled example
            example = self.hard_examples[idx]
            return {
                "example": example,
                "is_hard": True,
                "label": example.correct_index,
            }
        else:
            # Soft-labeled example
            soft_idx = idx - len(self.hard_examples)
            example = self.soft_examples[soft_idx]
            return {
                "example": example,
                "is_hard": False,
                "soft_probs": self.soft_probs[soft_idx],
            }


class ActiveLoopExperiment(BaseExperiment):
    """Experiment for active learning loop with teacher distillation."""
    
    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        """
        Initialize active loop experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.model.device)
        
        if config.active is None:
            raise ValueError("ActiveLoopConfig is required for ActiveLoopExperiment")
        if config.kd is None:
            raise ValueError("KDConfig is required for ActiveLoopExperiment")
        
        self.active_config = config.active
        self.kd_config = config.kd
        
        # Setup logging
        self.logger = setup_logging(
            log_dir="outputs/logs",
            experiment_name=config.experiment_name,
        )
        
        # Create output directory
        self.output_dir = Path(config.training.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data splits
        self.logger.info("Loading dataset splits...")
        self.splits = load_splits(
            config.dataset,
            unlabeled_pool_size=config.active.unlabeled_pool_size,
        )
        
        # Create dataloaders for initial training
        self.logger.info("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            dataset_cfg=config.dataset,
            model_name=config.model.student_model_name,
            training_cfg=config.training,
            unlabeled_pool_size=config.active.unlabeled_pool_size,
        )
        
        # Build models
        self.logger.info(f"Building student model: {config.model.student_model_name}")
        self.student_model = build_student_model(config.model)
        self.student_model.to(self.device)
        
        self.logger.info(f"Building teacher ensemble: {config.model.teacher_model_names}")
        self.teacher_ensemble = build_teacher_ensemble(config.model)
        self.teacher_ensemble.to(self.device)
        self.teacher_ensemble.eval()
        
        # Store baseline metrics
        self.baseline_metrics: Dict[str, float] = {}
    
    def train(self) -> None:
        """Train the model through active learning loop."""
        self.logger.info("Starting active learning loop...")
        
        # Step 1: Train Student_v0 on labeled data
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Training Student_v0 on labeled data")
        self.logger.info("=" * 60)
        
        student_v0 = self._train_student_v0()
        
        # Step 2: Compute uncertainty on unlabeled pool
        self.logger.info("=" * 60)
        self.logger.info("Step 2: Computing uncertainty on unlabeled pool")
        self.logger.info("=" * 60)
        
        selected_indices = self._select_uncertain_examples(student_v0)
        
        # Step 3: Get teacher soft labels for selected examples
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Getting teacher soft labels for selected examples")
        self.logger.info("=" * 60)
        
        soft_examples, soft_probs = self._get_teacher_labels(selected_indices)
        
        # Step 4: Train Student_v1 with combined dataset
        self.logger.info("=" * 60)
        self.logger.info("Step 4: Training Student_v1 with combined dataset")
        self.logger.info("=" * 60)
        
        self._train_student_v1(soft_examples, soft_probs)
        
        self.logger.info("Active learning loop completed!")
    
    def _train_student_v0(self) -> torch.nn.Module:
        """
        Train initial student model on labeled data.
        
        Returns:
            Trained student model
        """
        # Create a temporary supervised experiment
        supervised_config = ExperimentConfig(
            experiment_name=f"{self.config.experiment_name}_v0",
            experiment_type="supervised",
            dataset=self.config.dataset,
            model=self.config.model,
            training=self.config.training,
            kd=None,
            active=None,
        )
        
        supervised_exp = SupervisedExperiment(supervised_config)
        
        # Train
        supervised_exp.train()
        
        # Get baseline metrics
        if "dev" in supervised_exp.dataloaders:
            self.baseline_metrics = supervised_exp._evaluate_on_loader(
                supervised_exp.dataloaders["dev"],
                split="baseline/dev",
            )
        
        # Return the trained model
        return supervised_exp.model
    
    def _select_uncertain_examples(
        self,
        student_model: torch.nn.Module,
    ) -> list[int]:
        """
        Select most uncertain examples from unlabeled pool.
        
        Args:
            student_model: Trained student model for uncertainty computation
        
        Returns:
            List of indices of selected examples
        """
        pool_loader = self.dataloaders.get("pool_unlabeled")
        if pool_loader is None:
            raise ValueError("pool_unlabeled dataloader not found")
        
        student_model.eval()
        
        all_probs = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(pool_loader, desc="Computing uncertainty")):
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                # Get student predictions
                logits = student_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                
                # Store batch indices
                start_idx = batch_idx * batch_size
                all_indices.extend(range(start_idx, start_idx + batch_size))
        
        # Concatenate all probabilities
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Compute uncertainty scores
        if self.active_config.uncertainty_metric == "entropy":
            uncertainty_scores = entropy(all_probs)
        elif self.active_config.uncertainty_metric == "margin":
            # Margin = 1 - max_prob
            max_probs = max_prob(all_probs)
            uncertainty_scores = 1.0 - max_probs
        else:
            raise ValueError(
                f"Unknown uncertainty metric: {self.active_config.uncertainty_metric}"
            )
        
        # Select top-K most uncertain examples
        top_k = min(self.active_config.top_k_uncertain, len(uncertainty_scores))
        top_indices = np.argsort(uncertainty_scores)[-top_k:][::-1]  # Sort descending
        
        # Map to actual example indices
        selected_indices = [all_indices[idx] for idx in top_indices]
        
        self.logger.info(
            f"Selected {len(selected_indices)} most uncertain examples "
            f"(uncertainty metric: {self.active_config.uncertainty_metric})"
        )
        
        return selected_indices
    
    def _get_teacher_labels(
        self,
        selected_indices: list[int],
    ) -> tuple[list[MCQAExample], list[torch.Tensor]]:
        """
        Get teacher soft labels for selected examples.
        
        Args:
            selected_indices: Indices of selected examples in pool_unlabeled
        
        Returns:
            Tuple of (selected examples, teacher soft label probabilities)
        """
        pool_examples = self.splits["pool_unlabeled"]
        selected_examples = [pool_examples[idx] for idx in selected_indices]
        
        # Create a dataset for selected examples
        selected_dataset = MultipleChoiceDataset(selected_examples)
        
        # Create a dataloader (no shuffle, batch size 1 to preserve order)
        selected_loader = DataLoader(
            selected_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.dataloaders["pool_unlabeled"].collate_fn,
        )
        
        soft_probs_list = []
        
        with torch.no_grad():
            for batch in tqdm(selected_loader, desc="Getting teacher labels"):
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                # Get teacher probabilities
                teacher_probs = self.teacher_ensemble.predict_proba(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                    temperature=self.kd_config.temperature,
                )
                
                soft_probs_list.append(teacher_probs.cpu())
        
        return selected_examples, soft_probs_list
    
    def _train_student_v1(
        self,
        soft_examples: list[MCQAExample],
        soft_probs: list[torch.Tensor],
    ) -> None:
        """
        Train Student_v1 with combined hard and soft labels.
        
        Args:
            soft_examples: Selected examples with soft labels
            soft_probs: Teacher probabilities for selected examples
        """
        # Create combined dataset
        hard_examples = self.splits["train_labeled"]
        combined_dataset = CombinedDataset(hard_examples, soft_examples, soft_probs)
        
        # Create dataloader with collator
        collator = self.dataloaders["train_labeled"].collate_fn
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=self._combined_collate_fn,
        )
        
        # Setup optimizer
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Setup scheduler
        warmup_steps = self.config.training.warmup_steps
        if warmup_steps > 0:
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            scheduler = None
        
        self.student_model.train()
        dev_loader = self.dataloaders.get("dev")
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            total_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(combined_loader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                is_hard = batch["is_hard"]
                labels = batch.get("labels")
                soft_probs_batch = batch.get("soft_probs")
                
                # Forward pass
                student_logits = self.student_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                
                # Compute loss
                loss = self._compute_combined_loss(
                    student_logits=student_logits,
                    is_hard=is_hard,
                    labels=labels,
                    soft_probs=soft_probs_batch,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.training.max_grad_norm,
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None and batch_idx < warmup_steps:
                    scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Evaluate on dev
            if dev_loader is not None:
                dev_metrics = self._evaluate_on_loader(dev_loader, split="dev")
                log_metrics(
                    step=f"train/epoch_{epoch + 1}",
                    metrics={"loss": avg_loss, **dev_metrics},
                    logger=self.logger,
                )
        
        # Save final model
        self.logger.info("Saving final model...")
        save_model(
            model=self.student_model,
            output_path=self.output_dir,
            config=self.config,
        )
    
    def _combined_collate_fn(
        self,
        batch: list[dict],
    ) -> dict[str, torch.Tensor]:
        """
        Custom collate function for combined dataset.
        
        Args:
            batch: List of items from CombinedDataset
        
        Returns:
            Collated batch
        """
        # Separate hard and soft examples
        hard_items = [item for item in batch if item["is_hard"]]
        soft_items = [item for item in batch if not item["is_hard"]]
        
        # Use original collator for hard examples
        if hard_items:
            hard_examples = [item["example"] for item in hard_items]
            hard_batch = self.dataloaders["train_labeled"].collate_fn(hard_examples)
        else:
            hard_batch = None
        
        # Use original collator for soft examples
        if soft_items:
            soft_examples = [item["example"] for item in soft_items]
            soft_batch = self.dataloaders["train_labeled"].collate_fn(soft_examples)
            # Add soft probabilities
            soft_probs = torch.stack([item["soft_probs"] for item in soft_items])
            soft_batch["soft_probs"] = soft_probs
            soft_batch["is_hard"] = torch.zeros(len(soft_items), dtype=torch.bool)
        else:
            soft_batch = None
        
        # Combine batches
        if hard_batch is None:
            return soft_batch
        elif soft_batch is None:
            hard_batch["is_hard"] = torch.ones(len(hard_items), dtype=torch.bool)
            return hard_batch
        else:
            # Combine both
            combined = {}
            for key in hard_batch.keys():
                if key == "is_hard":
                    combined[key] = torch.cat([hard_batch[key], soft_batch[key]])
                elif key == "soft_probs":
                    combined[key] = soft_batch[key]
                else:
                    combined[key] = torch.cat([hard_batch[key], soft_batch[key]])
            
            # Update batch_size
            combined["batch_size"] = len(batch)
            return combined
    
    def _compute_combined_loss(
        self,
        student_logits: torch.Tensor,
        is_hard: torch.Tensor,
        labels: torch.Tensor | None,
        soft_probs: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute combined loss for hard and soft labeled examples.
        
        Args:
            student_logits: Student logits, shape (batch_size, num_options)
            is_hard: Boolean tensor indicating hard labels, shape (batch_size,)
            labels: Hard labels, shape (batch_size,) or None
            soft_probs: Soft label probabilities, shape (batch_size, num_options) or None
        
        Returns:
            Combined loss
        """
        temperature = self.kd_config.temperature
        alpha = self.kd_config.alpha
        
        losses = []
        
        # Process hard-labeled examples
        hard_mask = is_hard.bool()
        if hard_mask.any() and labels is not None:
            hard_logits = student_logits[hard_mask]
            hard_labels = labels[hard_mask]
            ce_loss = F.cross_entropy(hard_logits, hard_labels)
            losses.append((1 - alpha) * ce_loss)
        
        # Process soft-labeled examples
        soft_mask = ~is_hard.bool()
        if soft_mask.any() and soft_probs is not None:
            soft_logits = student_logits[soft_mask]
            soft_probs_masked = soft_probs[soft_mask].to(self.device)
            
            # KD loss
            student_logits_T = soft_logits / temperature
            teacher_probs_T = soft_probs_masked
            
            student_probs_T = F.log_softmax(student_logits_T, dim=-1)
            
            kd_loss = F.kl_div(
                student_probs_T,
                teacher_probs_T,
                reduction="batchmean",
            )
            kd_loss = kd_loss * (temperature ** 2)
            
            losses.append(alpha * kd_loss)
        
        # Also apply KD loss to hard examples if alpha > 0
        if hard_mask.any() and labels is not None and alpha > 0:
            # For hard examples, we can also use KD with teacher
            # This is optional but can help
            pass  # Skip for simplicity, can be added later
        
        if not losses:
            raise ValueError("No examples to compute loss on")
        
        return sum(losses)
    
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
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                logits = self.student_model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
        
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        acc = accuracy(all_predictions, all_labels)
        exp_correct = expected_correctness(all_probs, all_labels)
        
        metrics = {
            "accuracy": acc,
            "expected_correctness": exp_correct,
        }
        
        self.student_model.train()
        return metrics
