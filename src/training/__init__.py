"""Training modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

__all__ = ["BaseExperiment"]


class BaseExperiment(ABC):
    """Abstract base class for all experiments."""
    
    @abstractmethod
    def train(self) -> None:
        """
        Train the model.
        
        This method should implement the full training loop.
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        pass
