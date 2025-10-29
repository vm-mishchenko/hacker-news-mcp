"""
Generate candidate datasets for manual review using multiple retrieval strategies and LLM ranking.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.packages.candidate_generator import CandidateGenerator
from src.evaluation.packages.candidate_ranking_service import CandidateRankingService
from src.evaluation.packages.llm_ranker import LLMRanker
from src.evaluation.packages.models import RankingResult
from src.packages.embedding_service import OpenAIEmbeddingService
from src.packages.mongodb_client import MongoDBClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "llm_model": "gpt-4o",
    "llm_temperature": 0.0,
    "database_name": "hacker-news",
    "collection_name": "posts",
}

QUERY = "mongodb"


class JSONResultWriter:
    """Write results to formatted JSON file."""

    def write_results(self, query: str, result: RankingResult, output_path: str):
        """Write results to a formatted JSON file."""
        logger.info(f"Writing results to {output_path}")

        # Build relevance template
        relevance_data = self._build_relevance_template(query, result.ranked_candidates)

        # Build complete JSON structure
        output_data = {
            "query": query,
            "ranked_candidates": result.ranked_candidates,
            "candidates_per_strategy": result.candidates_per_strategy,
            "all_candidates": result.all_candidates,
            "llm_input_candidates": result.llm_input_candidates,
            "relevance": relevance_data
        }

        # Write to file (one line per top-level key)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('{\n')
            keys = list(output_data.keys())
            for i, key in enumerate(keys):
                f.write(f'  "{key}": {json.dumps(output_data[key], default=str)}')
                if i < len(keys) - 1:
                    f.write(',\n')
                else:
                    f.write('\n')
            f.write('}\n')

        logger.info(f"Results written to {output_path}")

    def _build_relevance_template(self, query: str, ranked_candidates: List[dict]) -> dict:
        """Build relevance dataset template from ranked candidates."""
        logger.info("Building relevance dataset template")

        doc_ids: List[str] = [str(doc['id']) for doc in ranked_candidates]
        relevance_data = {
            "query": query,
            "relevant_docs": doc_ids
        }

        logger.info(f"Relevance template built with {len(doc_ids)} document IDs")
        return relevance_data


def _print_summary(query: str, ranked_candidates: List[dict]):
    """Print query and top 10 ranked titles with score and date."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Query: {query}")
    logger.info("")
    logger.info("Top 10 Ranked Titles:")

    for i, candidate in enumerate(ranked_candidates[:10], 1):
        post_id = candidate.get('id', 'N/A')
        title = candidate.get('title', 'N/A')
        score = candidate.get('score', 0)
        time_obj = candidate.get('time')

        # Extract year and month from datetime object
        if time_obj:
            year_month = time_obj.strftime('%Y-%m')
        else:
            year_month = 'N/A'

        logger.info(f"{i}. {title} (id: {post_id}, score: {score}, date: {year_month})")

    logger.info("=" * 80 + "\n")


def run():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate candidate datasets for manual review using multiple retrieval strategies and LLM ranking.'
    )
    parser.add_argument(
        '--query',
        type=str,
        default=QUERY,
        help='Search query to generate candidates for (default: mongodb)'
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=20,
        help='Maximum number of candidates to send to LLM for ranking (default: 20)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=30,
        help='Number of results to retrieve per retrieval strategy (default: 30)'
    )
    args = parser.parse_args()
    query = args.query
    max_candidates = args.max_candidates
    top_k_per_strategy = args.top_k

    logger.info(f"Processing query: {query}")
    logger.info(f"Max candidates for LLM ranking: {max_candidates}")
    logger.info(f"Top-k per strategy: {top_k_per_strategy}")

    # Load .env.local from project root (must run from project root)
    env_path = Path('.env.local')
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env.local not found at {env_path}")

    # Initialize MongoDB client
    mongodb_username = os.getenv('MONGODB_USERNAME')
    mongodb_password = os.getenv('MONGODB_PASSWORD')
    mongodb_uri = os.getenv('MONGODB_URI')

    if not all([mongodb_username, mongodb_password, mongodb_uri]):
        logger.error("Missing MongoDB credentials in environment variables")
        return

    mongodb_client = MongoDBClient(
        username=mongodb_username,
        password=mongodb_password,
        uri=mongodb_uri
    )
    mongo_client = mongodb_client.get_client()
    logger.info("MongoDB client initialized")

    # Initialize OpenAI services
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("Missing OPENAI_API_KEY in environment variables")
        return

    embedding_service = OpenAIEmbeddingService(api_key=openai_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI services initialized")

    # Initialize components
    candidate_generator = CandidateGenerator(
        mongo_client=mongo_client,
        database_name=CONFIG["database_name"],
        collection_name=CONFIG["collection_name"],
        embedding_service=embedding_service
    )

    llm_ranker = LLMRanker(
        openai_client=openai_client,
        model=CONFIG["llm_model"],
        temperature=CONFIG["llm_temperature"]
    )

    ranking_service = CandidateRankingService(
        candidate_generator=candidate_generator,
        llm_ranker=llm_ranker
    )

    # Generate and rank candidates
    result = ranking_service.get_ranked_candidates(
        query=query,
        top_k=top_k_per_strategy,
        max_candidates=max_candidates
    )

    logger.info(f"LLM ranked {len(result.ranked_candidates)} candidates")
    logger.info(f"LLM received {len(result.llm_input_candidates)} candidates for ranking")

    # Save results
    json_writer = JSONResultWriter()
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_path = f"evaluation/candidates_ranked_{timestamp}.json"
    json_writer.write_results(
        query=query,
        result=result,
        output_path=str(output_path)
    )

    logger.info(f"Results saved to {output_path}")

    # Print summary
    _print_summary(query, result.ranked_candidates)

    logger.info("Candidate generation complete")


if __name__ == "__main__":
    run()
