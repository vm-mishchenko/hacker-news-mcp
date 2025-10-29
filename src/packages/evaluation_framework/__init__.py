"""
Evaluation Framework for Search/Retrieval Systems

A minimal, file-based evaluation framework for iterative search system improvement.
Designed to be project-agnostic and reusable.
"""

from .dataset import Dataset
from .evaluator import Evaluator
from .experiment import save_experiment, load_experiment, compare_experiments
from .models import DocumentId, QueryText, QueryResult, Metrics, ExperimentConfig, Experiment
from .retriever import Retriever

__all__ = [
    "DocumentId",
    "QueryText",
    "QueryResult",
    "Metrics",
    "ExperimentConfig",
    "Experiment",
    "Dataset",
    "Retriever",
    "Evaluator",
    "save_experiment",
    "load_experiment",
    "compare_experiments",
]
