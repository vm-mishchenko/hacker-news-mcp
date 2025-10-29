"""
Generate relevance dataset from multiple queries using retrieval strategies and LLM ranking.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.packages.candidate_generator import CandidateGenerator
from src.evaluation.packages.candidate_ranking_service import CandidateRankingService
from src.evaluation.packages.llm_ranker import LLMRanker
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


def _read_queries(file_path: str) -> List[str]:
    """Read queries from JSONL file."""
    logger.info(f"Reading queries from {file_path}")

    queries: List[str] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                query = data['query']
                queries.append(query)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                raise

    logger.info(f"Read {len(queries)} queries from file")
    return queries


def _write_entry(file_handle, query: str, doc_ids: List[str]) -> None:
    """Write single relevance dataset entry to file."""
    entry = {
        "query": query,
        "relevant_docs": doc_ids
    }
    file_handle.write(json.dumps(entry) + '\n')
    file_handle.flush()  # Ensure immediate write


def run():
    """Main coordinator function."""
    logger.info("Starting relevance dataset generation script")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate relevance dataset from multiple queries using retrieval strategies and LLM ranking.'
    )

    # Default queries file path relative to this script
    default_queries_file = 'src/evaluation/datasets/search_queries.jsonl'

    parser.add_argument(
        '--queries-file',
        type=str,
        default=str(default_queries_file),
        help=f'Path to JSONL file containing search queries (default: {default_queries_file})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N queries (optional, for testing)'
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
    queries_file = args.queries_file
    limit = args.limit
    max_candidates = args.max_candidates
    top_k_per_strategy = args.top_k

    # Read queries from file
    queries = _read_queries(queries_file)

    # Apply limit if specified
    if limit is not None:
        queries = queries[:limit]
        logger.info(f"Limited to first {limit} queries")

    total_queries = len(queries)
    logger.info(f"Will process {total_queries} queries")
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

    # Create output file
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_path = f"evaluation/candidate_relevance_dataset_{timestamp}.jsonl"
    logger.info(f"Output will be written to: {output_path}")

    # Process queries with timing
    start_time = time.time()

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, query in enumerate(queries, 1):
            logger.info(f"Processing query {idx}/{total_queries}: {query}")

            # Generate and rank candidates
            result = ranking_service.get_ranked_candidates(
                query=query,
                top_k=top_k_per_strategy,
                max_candidates=max_candidates
            )

            # Extract document IDs
            doc_ids: List[str] = [str(doc['id']) for doc in result.ranked_candidates]

            # Log warning if no results
            if len(doc_ids) == 0:
                logger.error(f"Query '{query}' returned 0 matching documents")

            # Write to file immediately
            _write_entry(f, query, doc_ids)
            logger.info(f"Wrote {len(doc_ids)} document IDs for query '{query}'")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log final summary
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total queries processed: {total_queries}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    logger.info("=" * 80)


if __name__ == "__main__":
    run()
