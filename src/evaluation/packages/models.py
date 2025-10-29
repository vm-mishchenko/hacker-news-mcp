"""Data models for evaluation packages."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RankingResult:
    """Result of candidate ranking operation."""
    ranked_candidates: List[dict]
    llm_input_candidates: List[dict]
    candidates_per_strategy: Dict[str, dict]
    all_candidates: List[dict]

