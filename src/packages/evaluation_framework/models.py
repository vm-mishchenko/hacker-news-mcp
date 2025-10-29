"""
Data models for the evaluation framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, NewType, Any

# Type aliases to enforce type safety
DocumentId = NewType('DocumentId', str)
QueryText = NewType('QueryText', str)


@dataclass
class QueryResult:
    """Single query with associated document IDs (ground truth or retrieval results)."""
    query: QueryText
    relevant_docs: List[DocumentId]


@dataclass
class Metrics:
    """Evaluation metrics for one experiment."""
    precision_at_k: Dict[int, float]  # {1: 0.8, 5: 0.6, 10: 0.5}
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    per_query_scores: Dict[str, Dict[str, float]]  # For failure analysis


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    timestamp: datetime
    retriever: str
    name: str = ""
    retriever_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Container for experiment data."""
    dataset: 'Dataset'
    metrics: Metrics
    config: ExperimentConfig
