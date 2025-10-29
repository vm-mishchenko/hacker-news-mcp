"""Rank candidates using OpenAI LLM."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

# Load prompt template from file
RANKING_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "rank_candidates.md"
RANKING_PROMPT_TEMPLATE = RANKING_PROMPT_PATH.read_text()


class LLMRanker:
    """Rank candidates using OpenAI LLM."""

    def __init__(self, openai_client: OpenAI, model: str, temperature: float):
        """Initialize LLM ranker."""
        self.client = openai_client
        self.model = model
        self.temperature = temperature

    def rank_candidates(self, query: str, candidates: List[dict], max_candidates: int = 20) -> \
        tuple[List[dict], List[dict]]:
        """Rank candidates using OpenAI Chat Completions API."""
        logger.info(f"Ranking {len(candidates)} candidates for query: '{query}'")

        # Limit candidates to max_candidates (sorted by HN score first)
        llm_input_candidates = candidates
        if len(candidates) > max_candidates:
            logger.info(
                f"Limiting candidates from {len(candidates)} to {max_candidates} (sorted by HN score)")
            llm_input_candidates = sorted(candidates, key=lambda x: x.get('score', 0),
                                          reverse=True)[
                :max_candidates]
            logger.info(f"Limited to {len(llm_input_candidates)} candidates")

        prompt = self._build_ranking_prompt(query, llm_input_candidates)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a search relevance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            llm_output = json.loads(response.choices[0].message.content)
            ranked_doc_ids = llm_output.get("ranked_ids", [])

            # Validate that LLM returned the same number of IDs as input
            if len(ranked_doc_ids) != len(llm_input_candidates):
                error_msg = (
                    f"LLM returned {len(ranked_doc_ids)} ranked IDs but received "
                    f"{len(llm_input_candidates)} input candidates. Expected equal counts."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate that all input IDs match all output IDs
            input_ids = set(str(c['id']) for c in llm_input_candidates)
            output_ids = set(str(doc_id) for doc_id in ranked_doc_ids)

            if input_ids != output_ids:
                missing_in_output = input_ids - output_ids
                extra_in_output = output_ids - input_ids
                error_msg = (
                    f"Input and output IDs do not match. "
                    f"Missing in output: {missing_in_output}, "
                    f"Extra in output: {extra_in_output}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Reorder candidates
            ranked_candidates = self._reorder_by_llm_ranking(llm_input_candidates, ranked_doc_ids)

            return ranked_candidates, llm_input_candidates
        except Exception as e:
            logger.error(f"LLM ranking failed: {e}")
            # Fallback: sort by HN score
            fallback_ranked = sorted(llm_input_candidates, key=lambda x: x.get('score', 0),
                                     reverse=True)
            return fallback_ranked, llm_input_candidates

    def _build_ranking_prompt(self, query: str, candidates: List[dict]) -> str:
        """Build prompt for LLM ranking."""
        logger.info("Building ranking prompt")

        # Format candidates as JSON, one per line
        candidates_json = "\n".join([
            json.dumps({
                "id": doc['id'],
                "title": doc['title'],
                "time": doc.get('time').strftime('%Y-%m-%d') if doc.get('time') else None
            })
            for doc in candidates
        ])

        # Use template from file
        prompt = RANKING_PROMPT_TEMPLATE.format(
            query=query,
            stories_json=candidates_json
        )

        logger.info("Ranking prompt built")
        return prompt

    def _reorder_by_llm_ranking(self, candidates: List[dict], ranked_ids: List[str]) -> List[dict]:
        """Reorder candidates based on LLM's ranked list of IDs."""
        logger.info(
            f"Reordering {len(candidates)} candidates based on {len(ranked_ids)} ranked IDs")

        # Create lookup map
        candidate_map = {str(c['id']): c for c in candidates}

        # Reorder and add rank field
        ranked_candidates: List[dict] = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in candidate_map:
                candidate = candidate_map[doc_id].copy()
                candidate['rank'] = rank
                ranked_candidates.append(candidate)

        logger.info(f"Reordered to {len(ranked_candidates)} ranked candidates")
        return ranked_candidates

    def _format_date(self, date_value) -> str:
        """Format date value to YYYY-MM-DD string."""
        if date_value is None:
            return 'N/A'
        if isinstance(date_value, datetime):
            return date_value.strftime('%Y-%m-%d')
        if isinstance(date_value, str):
            return date_value[:10]
        return 'N/A'
