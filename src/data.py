"""Dataset loading and processing for medical MCQA tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from src.config import DatasetConfig, TrainingConfig


@dataclass
class MCQAExample:
    """Normalized representation of a multiple choice question example."""
    
    question: str
    options: list[str]
    correct_index: int


class MultipleChoiceDataset(TorchDataset):
    """PyTorch Dataset for multiple choice examples."""
    
    def __init__(
        self,
        examples: list[MCQAExample],
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            examples: List of normalized examples
        """
        self.examples = examples
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.examples)
    
    def __getitem__(
        self,
        idx: int,
    ) -> MCQAExample:
        """Return an example by index."""
        return self.examples[idx]


class MultipleChoiceCollator:
    """Collator for tokenizing multiple choice examples."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        """
        Initialize the collator.
        
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(
        self,
        batch: list[MCQAExample],
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize a batch of examples.
        
        Each example is expanded into num_options inputs (question + option).
        Returns a flat batch of size (batch_size * num_options).
        
        Args:
            batch: List of MCQA examples
        
        Returns:
            Dictionary with tokenized data:
            - input_ids: (batch_size * num_options, seq_len)
            - attention_mask: (batch_size * num_options, seq_len)
            - labels: (batch_size,) - indices of correct answers
            - num_options: (batch_size,) - number of options for each example
        """
        num_options_list = [len(ex.options) for ex in batch]
        max_num_options = max(num_options_list)
        
        # Check that all examples have the same number of options
        if len(set(num_options_list)) > 1:
            raise ValueError(
                f"All examples must have the same number of options. "
                f"Got: {num_options_list}"
            )
        
        num_options = num_options_list[0]
        batch_size = len(batch)
        
        # Create a flat list of (question + option) pairs
        flat_inputs = []
        labels = []
        
        for example in batch:
            for idx, option in enumerate(example.options):
                # Format input as "question [SEP] option" or simple concatenation
                sep_token = self.tokenizer.sep_token or self.tokenizer.eos_token or " "
                text = f"{example.question} {sep_token} {option}"
                flat_inputs.append(text)
            
            labels.append(example.correct_index)
        
        # Tokenize all inputs
        tokenized = self.tokenizer(
            flat_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_options": torch.tensor([num_options] * batch_size, dtype=torch.long),
            "batch_size": batch_size,
        }


def normalize_medmcqa_example(
    example: dict,
) -> MCQAExample:
    """
    Normalize an example from MedMCQA dataset.
    
    Args:
        example: Example from HuggingFace dataset
    
    Returns:
        Normalized MCQA example
    """
    question = example.get("question", "")
    options = [
        example.get("opa", ""),
        example.get("opb", ""),
        example.get("opc", ""),
        example.get("opd", ""),
    ]
    
    # Try to get correct answer index
    # MedMCQA uses "cop" field (correct option), which can be 0, 1, 2, 3 or 1, 2, 3, 4
    # If cop is missing or None, try other fields
    correct_index = example.get("cop")
    
    # Handle None or missing values
    if correct_index is None:
        # Try alternative field names
        correct_index = example.get("correct", example.get("answer_idx", 0))
    
    # Normalize to 0-indexed (MedMCQA may use 1-indexed: 1,2,3,4 -> 0,1,2,3)
    # Check if index is out of range for 0-indexed (should be 0-3)
    if isinstance(correct_index, (int, float)) and correct_index > 3:
        # Assume 1-indexed and convert to 0-indexed
        correct_index = int(correct_index) - 1
    elif isinstance(correct_index, (int, float)):
        correct_index = int(correct_index)
    else:
        # If still not valid, default to 0
        correct_index = 0
    
    # Ensure valid range [0, 3]
    correct_index = max(0, min(3, correct_index))
    
    return MCQAExample(
        question=question,
        options=options,
        correct_index=correct_index,
    )


def normalize_medqa_example(
    example: dict,
) -> MCQAExample:
    """
    Normalize an example from MedQA dataset.
    
    Args:
        example: Example from HuggingFace dataset
    
    Returns:
        Normalized MCQA example
    """
    # MedQA may have a different structure, adapt to common format
    question = example.get("question", "")
    
    # Try to find answer options in different possible formats
    if "options" in example and isinstance(example["options"], list):
        options = example["options"]
    elif "choices" in example and isinstance(example["choices"], list):
        options = example["choices"]
    else:
        # Try to find opa, opb, opc, opd
        options = [
            example.get("opa", ""),
            example.get("opb", ""),
            example.get("opc", ""),
            example.get("opd", ""),
        ]
    
    # Find the correct answer
    if "answer_idx" in example:
        correct_index = example["answer_idx"]
    elif "correct" in example:
        correct_index = example["correct"]
    elif "cop" in example:
        correct_index = example["cop"]
    else:
        correct_index = 0
    
    return MCQAExample(
        question=question,
        options=options,
        correct_index=correct_index,
    )


def load_and_normalize_dataset(
    dataset_name: str,
    split: str,
) -> list[MCQAExample]:
    """
    Load and normalize a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Dataset name ("medqa" or "medmcqa")
        split: Dataset split (train, validation, test)
    
    Returns:
        List of normalized examples
    """
    # Load the dataset
    if dataset_name.lower() == "medmcqa":
        # Try to load with config "A", fallback to "default" if not available
        try:
            dataset = load_dataset("medmcqa", "A", split=split)
        except ValueError:
            # If config "A" is not available, use "default"
            dataset = load_dataset("medmcqa", split=split)
        normalize_fn = normalize_medmcqa_example
    elif dataset_name.lower() == "medqa":
        # MedQA may be available under different names
        try:
            dataset = load_dataset("bigbio/med_qa", split=split)
        except Exception:
            # Try alternative source
            dataset = load_dataset("medqa", split=split)
        normalize_fn = normalize_medqa_example
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported: 'medqa', 'medmcqa'"
        )
    
    # Normalize all examples
    normalized_examples = [normalize_fn(example) for example in dataset]
    
    return normalized_examples


def load_splits(
    config: DatasetConfig,
    unlabeled_pool_size: int = 1000,
) -> dict[str, list[MCQAExample]]:
    """
    Load and split the dataset into required parts.
    
    Args:
        config: Dataset configuration
        unlabeled_pool_size: Size of unlabeled data pool (selected from train)
    
    Returns:
        Dictionary with splits:
        - "train_labeled": Labeled examples for training
        - "dev": Development set
        - "test": Test set
        - "pool_unlabeled": Pool of unlabeled examples for active learning
    """
    # Load train split
    train_examples = load_and_normalize_dataset(
        config.dataset_name,
        config.train_split,
    )
    
    # Limit dataset size for quick testing
    if config.max_samples is not None and len(train_examples) > config.max_samples:
        import random
        random.seed(config.seed)
        random.shuffle(train_examples)
        train_examples = train_examples[:config.max_samples]
    
    # Load dev/validation split
    if config.val_split:
        dev_examples = load_and_normalize_dataset(
            config.dataset_name,
            config.val_split,
        )
        # Limit dev size for quick testing
        if config.max_samples is not None and len(dev_examples) > config.max_samples // 10:
            import random
            random.seed(config.seed)
            random.shuffle(dev_examples)
            dev_examples = dev_examples[:config.max_samples // 10]
    else:
        # If val_split is not specified, create from train
        import random
        random.seed(config.seed)
        random.shuffle(train_examples)
        split_idx = max(1, int(len(train_examples) * 0.1))
        dev_examples = train_examples[:split_idx]
        train_examples = train_examples[split_idx:]
    
    # Load test split
    if config.test_split:
        test_examples = load_and_normalize_dataset(
            config.dataset_name,
            config.test_split,
        )
        # Limit test size for quick testing
        if config.max_samples is not None and len(test_examples) > config.max_samples // 10:
            import random
            random.seed(config.seed)
            random.shuffle(test_examples)
            test_examples = test_examples[:config.max_samples // 10]
        
        # Debug: Check if test examples have valid labels
        # In MedMCQA, test split may not have labels (cop field may be missing)
        # Count examples with non-zero labels
        non_zero_labels = sum(1 for ex in test_examples if ex.correct_index != 0)
        if non_zero_labels == 0 and len(test_examples) > 0:
            # If all labels are 0, test split likely doesn't have ground truth
            # In this case, we should use validation split for evaluation instead
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Test split appears to have no ground truth labels (all labels are 0). "
                f"This is common in MedMCQA test split. Consider using validation split for evaluation."
            )
    else:
        test_examples = []
    
    # Create pool_unlabeled from train
    # For the first version, take a subset of train as unlabeled pool
    import random
    random.seed(config.seed)
    random.shuffle(train_examples)
    
    if len(train_examples) > unlabeled_pool_size:
        pool_unlabeled = train_examples[:unlabeled_pool_size]
        train_labeled = train_examples[unlabeled_pool_size:]
    else:
        # If train is smaller than unlabeled_pool_size, use all train
        pool_unlabeled = train_examples[:len(train_examples) // 2]
        train_labeled = train_examples[len(train_examples) // 2:]
    
    return {
        "train_labeled": train_labeled,
        "dev": dev_examples,
        "test": test_examples,
        "pool_unlabeled": pool_unlabeled,
    }


def create_dataloaders(
    dataset_cfg: DatasetConfig,
    model_name: str,
    training_cfg: TrainingConfig,
    unlabeled_pool_size: int = 1000,
) -> dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for all dataset splits.
    
    Args:
        dataset_cfg: Dataset configuration
        model_name: Model name (for loading tokenizer)
        training_cfg: Training configuration
        unlabeled_pool_size: Size of unlabeled data pool
    
    Returns:
        Dictionary with DataLoaders:
        - "train_labeled": DataLoader for training
        - "dev": DataLoader for validation
        - "test": DataLoader for testing
        - "pool_unlabeled": DataLoader for unlabeled pool
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # If tokenizer doesn't have sep_token, use eos_token
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
    
    # Create collator
    collator = MultipleChoiceCollator(
        tokenizer=tokenizer,
        max_length=dataset_cfg.max_seq_length,
    )
    
    # Load splits
    splits = load_splits(dataset_cfg, unlabeled_pool_size=unlabeled_pool_size)
    
    # Create DataLoaders
    dataloaders = {}
    
    # Train labeled
    train_dataset = MultipleChoiceDataset(splits["train_labeled"])
    dataloaders["train_labeled"] = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # For CPU
        pin_memory=False,
    )
    
    # Dev
    if splits["dev"]:
        dev_dataset = MultipleChoiceDataset(splits["dev"])
        dataloaders["dev"] = DataLoader(
            dev_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
    
    # Test
    if splits["test"]:
        test_dataset = MultipleChoiceDataset(splits["test"])
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
    
    # Pool unlabeled
    if splits["pool_unlabeled"]:
        pool_dataset = MultipleChoiceDataset(splits["pool_unlabeled"])
        dataloaders["pool_unlabeled"] = DataLoader(
            pool_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
    
    return dataloaders
