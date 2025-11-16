"""Active learning loop with teacher distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data import (
    MCQAExample,
    MultipleChoiceCollator,
    MultipleChoiceDataset,
    create_dataloaders,
    load_splits,
)
from src.metrics import accuracy, entropy, expected_correctness, max_prob
from src.models import build_student_model, build_teacher_ensemble
from src.training import BaseExperiment
from src.training.supervised import SupervisedExperiment
from src.utils import log_metrics, move_to_device, save_model, setup_logging


class CombinedDataset(Dataset):
    """Dataset combining labeled examples (with hard labels) and pseudo-labeled examples (with soft labels)."""
    
    def __init__(
        self,
        labeled_examples: list[MCQAExample],
        pseudo_labeled_examples: list[tuple[MCQAExample, torch.Tensor]],
    ) -> None:
        """
        Initialize combined dataset.
        
        Args:
            labeled_examples: Examples with hard labels
            pseudo_labeled_examples: Examples with teacher soft labels (probabilities)
        """
        self.labeled_examples = labeled_examples
        self.pseudo_labeled_examples = pseudo_labeled_examples
        self.is_labeled = [True] * len(labeled_examples) + [False] * len(pseudo_labeled_examples)
    
    def __len__(self) -> int:
        """Return total number of examples."""
        return len(self.labeled_examples) + len(self.pseudo_labeled_examples)
    
    def __getitem__(self, idx: int) -> tuple[MCQAExample, torch.Tensor | None, torch.Tensor | None, bool]:
        """
        Get item by index.
        
        Returns:
            Tuple of (example, hard_label, soft_label, is_labeled)
        """
        if idx < len(self.labeled_examples):
            example = self.labeled_examples[idx]
            return (example, torch.tensor(example.correct_index, dtype=torch.long), None, True)
        else:
            idx -= len(self.labeled_examples)
            example, soft_label = self.pseudo_labeled_examples[idx]
            return (example, None, soft_label, False)


class CombinedCollator:
    """Collator for combined dataset with hard and soft labels."""
    
    def __init__(
        self,
        base_collator: MultipleChoiceCollator,
    ) -> None:
        """
        Initialize combined collator.
        
        Args:
            base_collator: Base collator for tokenization
        """
        self.base_collator = base_collator
    
    def __call__(
        self,
        batch: list[tuple[MCQAExample, torch.Tensor | None, torch.Tensor | None, bool]],
    ) -> dict[str, torch.Tensor]:
        """
        Collate batch with mixed hard and soft labels.
        
        Args:
            batch: List of (example, hard_label, soft_label, is_labeled) tuples
        
        Returns:
            Dictionary with tokenized data and labels
        """
        examples = [item[0] for item in batch]
        hard_labels = [item[1] for item in batch]
        soft_labels = [item[2] for item in batch]
        is_labeled = [item[3] for item in batch]
        
        # Use base collator for tokenization
        base_batch = self.base_collator(examples)
        
        # Add hard and soft labels
        hard_labels_tensor = torch.stack([label for label in hard_labels if label is not None])
        soft_labels_list = [label for label in soft_labels if label is not None]
        if soft_labels_list:
            soft_labels_tensor = torch.stack(soft_labels_list)
        else:
            soft_labels_tensor = None
        
        is_labeled_tensor = torch.tensor(is_labeled, dtype=torch.bool)
        
        return {
            **base_batch,
            "hard_labels": hard_labels_tensor,
            "soft_labels": soft_labels_tensor,
            "is_labeled": is_labeled_tensor,
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
        self.logger.info("Loading dataset and creating splits...")
        self.splits = load_splits(
            config.dataset,
            unlabeled_pool_size=self.active_config.unlabeled_pool_size,
        )
        
        # Create dataloaders for initial training
        self.logger.info("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            dataset_cfg=config.dataset,
            model_name=config.model.student_model_name,
            training_cfg=config.training,
            unlabeled_pool_size=self.active_config.unlabeled_pool_size,
        )
        
        # Build teacher ensemble
        self.logger.info(f"Building teacher ensemble: {config.model.teacher_model_names}")
        self.teacher_ensemble = build_teacher_ensemble(config.model)
        self.teacher_ensemble.to(self.device)
        self.teacher_ensemble.eval()
        
        # Student models will be created during training
        self.student_v0 = None
        self.student_v1 = None
        
        self.logger.info(f"Initialized ActiveLoop experiment: {config.experiment_name}")
        self.logger.info(f"Train labeled: {len(self.splits['train_labeled'])}")
        self.logger.info(f"Pool unlabeled: {len(self.splits['pool_unlabeled'])}")
        self.logger.info(f"Top-K uncertain: {self.active_config.top_k_uncertain}")
        self.logger.info(f"Uncertainty metric: {self.active_config.uncertainty_metric}")
        
        # Store training metrics
        self.training_metrics: list[dict[str, Any]] = []
    
    def train(self) -> None:
        """Train the model through active learning loop."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Active Learning Loop")
        self.logger.info("=" * 60)
        
        # Step 1: Train Student_v0 on labeled data
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 1: Training Student_v0 on labeled data")
        self.logger.info("=" * 60)
        
        student_v0_config = ExperimentConfig(
            experiment_name=f"{self.config.experiment_name}_v0",
            experiment_type="supervised",
            dataset=self.config.dataset,
            model=self.config.model,
            training=self.config.training,
            kd=None,
            active=None,
        )
        
        supervised_exp = SupervisedExperiment(student_v0_config)
        supervised_exp.train()
        
        # Load Student_v0
        self.student_v0 = supervised_exp.model
        self.student_v0.eval()
        
        # Evaluate Student_v0
        if "dev" in self.dataloaders:
            v0_metrics = self._evaluate_student(self.student_v0, self.dataloaders["dev"], "Student_v0/dev")
            self.logger.info(f"Student_v0 dev metrics: {v0_metrics}")
        
        # Step 2: Active selection - compute uncertainty on pool_unlabeled
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 2: Active selection from unlabeled pool")
        self.logger.info("=" * 60)
        
        selected_indices = self._select_uncertain_examples()
        self.logger.info(f"Selected {len(selected_indices)} examples from unlabeled pool")
        
        # Step 3: Teacher querying - get soft labels for selected examples
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 3: Querying teacher ensemble for soft labels")
        self.logger.info("=" * 60)
        
        pseudo_labeled_examples = self._get_teacher_labels(selected_indices)
        self.logger.info(f"Obtained teacher soft labels for {len(pseudo_labeled_examples)} examples")
        
        # Step 4: Retrain Student_v1 on combined dataset
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Step 4: Training Student_v1 on combined dataset")
        self.logger.info("=" * 60)
        
        self._train_student_v1(pseudo_labeled_examples)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Active Learning Loop completed")
        self.logger.info("=" * 60)
    
    def _select_uncertain_examples(self) -> list[int]:
        """
        Select most uncertain examples from unlabeled pool.
        
        Returns:
            List of indices of selected examples
        """
        pool_loader = self.dataloaders["pool_unlabeled"]
        self.student_v0.eval()
        
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
                logits = self.student_v0.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                probs = F.softmax(logits, dim=-1)
                
                all_probs.append(probs.cpu().numpy())
                # Compute indices in the original pool
                start_idx = batch_idx * batch_size
                all_indices.extend(range(start_idx, start_idx + batch_size))
        
        # Concatenate all probabilities
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Compute uncertainty scores
        if self.active_config.uncertainty_metric == "entropy":
            uncertainty_scores = entropy(all_probs)
        elif self.active_config.uncertainty_metric == "margin":
            # Margin = 1 - max_prob (uncertainty)
            uncertainty_scores = 1.0 - max_prob(all_probs)
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.active_config.uncertainty_metric}")
        
        # Select top-K most uncertain
        top_k = min(self.active_config.top_k_uncertain, len(uncertainty_scores))
        top_indices = np.argsort(uncertainty_scores)[-top_k:][::-1]  # Sort descending
        
        # Map back to original pool indices
        selected_indices = [all_indices[idx] for idx in top_indices]
        
        return selected_indices
    
    def _get_teacher_labels(
        self,
        selected_indices: list[int],
    ) -> list[tuple[MCQAExample, torch.Tensor]]:
        """
        Get teacher ensemble soft labels for selected examples.
        
        Args:
            selected_indices: Indices of selected examples in pool_unlabeled
        
        Returns:
            List of (example, teacher_probs) tuples
        """
        selected_examples = [self.splits["pool_unlabeled"][idx] for idx in selected_indices]
        
        # Create a dataset and dataloader for selected examples
        selected_dataset = MultipleChoiceDataset(selected_examples)
        
        # Get tokenizer and create collator
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.student_model_name)
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token
        
        collator = MultipleChoiceCollator(
            tokenizer=tokenizer,
            max_length=self.config.dataset.max_seq_length,
        )
        
        selected_loader = DataLoader(
            selected_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
        
        pseudo_labeled = []
        example_idx = 0
        
        self.teacher_ensemble.eval()
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
                
                # Get corresponding examples from selected_examples
                for i in range(batch_size):
                    if example_idx < len(selected_examples):
                        example = selected_examples[example_idx]
                        probs = teacher_probs[i]
                        pseudo_labeled.append((example, probs.cpu()))
                        example_idx += 1
        
        return pseudo_labeled
    
    def _train_student_v1(
        self,
        pseudo_labeled_examples: list[tuple[MCQAExample, torch.Tensor]],
    ) -> None:
        """
        Train Student_v1 on combined dataset (labeled + pseudo-labeled).
        
        Args:
            pseudo_labeled_examples: Examples with teacher soft labels
        """
        # Create combined dataset
        combined_dataset = CombinedDataset(
            labeled_examples=self.splits["train_labeled"],
            pseudo_labeled_examples=pseudo_labeled_examples,
        )
        
        # Create collator
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.student_model_name)
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token
        
        base_collator = MultipleChoiceCollator(
            tokenizer=tokenizer,
            max_length=self.config.dataset.max_seq_length,
        )
        combined_collator = CombinedCollator(base_collator)
        
        # Create dataloader
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=combined_collator,
            num_workers=0,
            pin_memory=False,
        )
        
        # Build Student_v1
        self.student_v1 = build_student_model(self.config.model)
        self.student_v1.to(self.device)
        self.student_v1.train()
        
        # Setup optimizer
        optimizer = AdamW(
            self.student_v1.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Training Student_v1 - Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            total_loss = 0.0
            num_batches = 0
            
            for batch in tqdm(combined_loader, desc=f"Epoch {epoch + 1}"):
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                hard_labels = batch["hard_labels"]
                soft_labels = batch["soft_labels"]
                is_labeled = batch["is_labeled"]
                
                # Forward pass
                student_logits = self.student_v1.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_options=num_options,
                    batch_size=batch_size,
                )
                
                # Compute loss
                loss = self._compute_combined_loss(
                    student_logits=student_logits,
                    hard_labels=hard_labels,
                    soft_labels=soft_labels,
                    is_labeled=is_labeled,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_v1.parameters(),
                        self.config.training.max_grad_norm,
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Evaluate on dev
            if "dev" in self.dataloaders:
                dev_metrics = self._evaluate_student(self.student_v1, self.dataloaders["dev"], f"Student_v1/dev_epoch_{epoch + 1}")
                epoch_metrics = {"loss": avg_loss, **dev_metrics}
                log_metrics(
                    step=f"train_v1/epoch_{epoch + 1}",
                    metrics=epoch_metrics,
                    logger=self.logger,
                )
            else:
                epoch_metrics = {"loss": avg_loss}
            
            # Store training metrics
            self.training_metrics.append({
                "epoch": epoch + 1,
                "stage": "student_v1",
                **epoch_metrics,
            })
        
        # Save Student_v1
        self.logger.info("Saving Student_v1...")
        save_model(
            model=self.student_v1,
            output_path=self.output_dir / "student_v1",
            config=self.config,
        )
    
    def _compute_combined_loss(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor | None,
        is_labeled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss for labeled and pseudo-labeled examples.
        
        Args:
            student_logits: Student logits, shape (batch_size, num_options)
            hard_labels: Hard labels for labeled examples (only for labeled ones)
            soft_labels: Soft labels (probabilities) for pseudo-labeled examples
            is_labeled: Boolean tensor indicating which examples are labeled
        
        Returns:
            Combined loss
        """
        labeled_mask = is_labeled
        unlabeled_mask = ~is_labeled
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Hard loss for labeled examples
        if labeled_mask.any():
            labeled_logits = student_logits[labeled_mask]
            labeled_hard = hard_labels  # Already filtered to only labeled examples
            ce_loss = F.cross_entropy(labeled_logits, labeled_hard)
            total_loss += (1 - self.kd_config.alpha) * ce_loss
        
        # KD loss for pseudo-labeled examples
        if soft_labels is not None and unlabeled_mask.any():
            # For pseudo-labeled examples, use teacher soft labels
            unlabeled_logits = student_logits[unlabeled_mask]
            unlabeled_soft = soft_labels  # Already filtered to only unlabeled examples
            
            # Apply temperature scaling
            temperature = self.kd_config.temperature
            student_logits_T = unlabeled_logits / temperature
            teacher_probs_T = unlabeled_soft / temperature
            
            student_probs_T = F.log_softmax(student_logits_T, dim=-1)
            teacher_probs_T = F.softmax(teacher_probs_T, dim=-1)
            
            kd_loss = F.kl_div(
                student_probs_T,
                teacher_probs_T,
                reduction="batchmean",
            )
            kd_loss = kd_loss * (temperature ** 2)
            
            total_loss += self.kd_config.alpha * kd_loss
        
        return total_loss
    
    def _evaluate_student(
        self,
        student_model: torch.nn.Module,
        loader: DataLoader,
        split_name: str,
    ) -> Dict[str, float]:
        """
        Evaluate a student model on a data loader.
        
        Args:
            student_model: Student model to evaluate
            loader: DataLoader to evaluate on
            split_name: Name of the split for logging
        
        Returns:
            Dictionary with metrics
        """
        student_model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
                batch = move_to_device(batch, self.device)
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                num_options = batch["num_options"]
                batch_size = int(batch["batch_size"].item()) if isinstance(batch["batch_size"], torch.Tensor) else batch["batch_size"]
                
                logits = student_model.forward(
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
        
        student_model.train()
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the final Student_v1 on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.student_v1 is None:
            raise ValueError("Student_v1 has not been trained yet. Call train() first.")
        
        if "test" not in self.dataloaders:
            self.logger.warning("No test set available for evaluation")
            return {}
        
        self.logger.info("Evaluating Student_v1 on test set...")
        test_metrics = self._evaluate_student(self.student_v1, self.dataloaders["test"], "Student_v1/test")
        
        log_metrics(
            step="eval/test",
            metrics=test_metrics,
            logger=self.logger,
        )
        
        return test_metrics
