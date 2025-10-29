"""Orchestrate candidate generation and ranking pipeline."""

import logging

from src.evaluation.packages.candidate_generator import CandidateGenerator
from src.evaluation.packages.llm_ranker import LLMRanker
from src.evaluation.packages.models import RankingResult

logger = logging.getLogger(__name__)


class CandidateRankingService:
    """Orchestrate candidate retrieval and ranking pipeline."""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        llm_ranker: LLMRanker
    ):
        """Initialize candidate ranking service."""
        self.candidate_generator = candidate_generator
        self.llm_ranker = llm_ranker

    def get_ranked_candidates(
        self,
        query: str,
        top_k: int,
        max_candidates: int
    ) -> RankingResult:
        """Generate and rank candidates for a query."""
        logger.info(f"Starting candidate ranking for query: '{query}'")

        # Generate candidates
        all_candidates, candidates_per_strategy = self.candidate_generator.generate_candidates(
            query, top_k)

        # Rank candidates
        ranked_candidates, llm_input_candidates = self.llm_ranker.rank_candidates(
            query, all_candidates, max_candidates
        )

        logger.info(f"Candidate ranking complete: {len(ranked_candidates)} ranked candidates")

        return RankingResult(
            ranked_candidates=ranked_candidates,
            llm_input_candidates=llm_input_candidates,
            candidates_per_strategy=candidates_per_strategy,
            all_candidates=all_candidates
        )
