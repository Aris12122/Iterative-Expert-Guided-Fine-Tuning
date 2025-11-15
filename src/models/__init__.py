"""Models for student and teacher."""

from src.config import ModelConfig
from src.models.student import StudentMCQAModel
from src.models.teachers import TeacherEnsemble, TeacherMCQAModel

__all__ = [
    "StudentMCQAModel",
    "TeacherMCQAModel",
    "TeacherEnsemble",
    "build_student_model",
    "build_teacher_ensemble",
]


def build_student_model(
    config: ModelConfig,
) -> StudentMCQAModel:
    """
    Build a student model from configuration.
    
    Args:
        config: Model configuration
    
    Returns:
        StudentMCQAModel instance
    """
    return StudentMCQAModel(config)


def build_teacher_ensemble(
    config: ModelConfig,
    weights: list[float] | None = None,
) -> TeacherEnsemble:
    """
    Build a teacher ensemble from configuration.
    
    Args:
        config: Model configuration
        weights: Ensemble weights for each teacher (if None, uniform weights)
    
    Returns:
        TeacherEnsemble instance
    """
    return TeacherEnsemble(config, weights=weights)
