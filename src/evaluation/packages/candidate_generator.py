"""Generate candidates using multiple retrieval strategies."""

import logging
from typing import List, Set

from pymongo import MongoClient

from src.packages.embedding_service import OpenAIEmbeddingService

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generate candidates using multiple retrieval strategies."""

    def __init__(
        self,
        mongo_client: MongoClient,
        database_name: str,
        collection_name: str,
        embedding_service: OpenAIEmbeddingService
    ):
        """Initialize candidate generator."""
        self.mongo_client = mongo_client
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.db = mongo_client[database_name]
        self.collection = self.db[collection_name]

    def generate_candidates(self, query: str, top_k: int) -> tuple[List[dict], dict]:
        """Generate candidates using 2 retrieval strategies with equal representation."""
        logger.info(f"Generating candidates for query: '{query}' with top_k={top_k}")

        # Calculate per-strategy limit to ensure equal representation
        per_strategy_limit = top_k // 2
        logger.info(
            f"Requesting {per_strategy_limit} documents per strategy for equal representation")

        # Run all 2 strategies with equal limits
        # Results are already sorted by relevance (BM25 score / vector similarity)
        bm25_ids = self._bm25_search(query, per_strategy_limit)
        semantic_ids = self._semantic_search(query, per_strategy_limit)

        # Ensure equal number of docs from each strategy
        # Take the minimum of what was requested vs what was returned
        actual_per_strategy = min(len(bm25_ids), len(semantic_ids), per_strategy_limit)
        logger.info(
            f"Taking top {actual_per_strategy} documents from each strategy (BM25 returned {len(bm25_ids)}, semantic returned {len(semantic_ids)})")

        # Take equal amounts from each strategy (already sorted by relevance)
        bm25_ids_balanced = bm25_ids[:actual_per_strategy]
        semantic_ids_balanced = semantic_ids[:actual_per_strategy]

        # Store strategy results with pipeline definitions and document IDs
        candidates_per_strategy = {
            "bm25": {
                "pipeline": self._get_bm25_pipeline(query, per_strategy_limit),
                "results": bm25_ids
            },
            "semantic": {
                "pipeline": self._get_semantic_pipeline(query, per_strategy_limit),
                "results": semantic_ids
            },
        }

        # Combine with equal representation from each strategy
        # Interleave results: pick one from BM25, one from semantic, repeat
        # This ensures most relevant docs from both searches are at the top
        combined_ids: List[int] = []
        interleave_count = min(len(bm25_ids_balanced), len(semantic_ids_balanced))
        for i in range(interleave_count):
            combined_ids.append(bm25_ids_balanced[i])
            combined_ids.append(semantic_ids_balanced[i])

        # Deduplicate (some docs might appear in both strategies)
        all_ids = self._deduplicate_candidates(combined_ids)

        # Fetch full documents
        candidates = self._fetch_documents(all_ids)
        logger.info(f"Fetched {len(candidates)} candidate documents")

        return candidates, candidates_per_strategy

    def _get_bm25_pipeline(self, query: str, top_k: int) -> List[dict]:
        """Get BM25 search pipeline definition."""
        return [
            {
                "$search": {
                    "index": "default_text",
                    "text": {
                        "query": query,
                        "fuzzy": {
                            "prefixLength": 1,
                            "maxEdits": 2,
                            "maxExpansions": 100
                        },
                        "path": "title"
                    }
                }
            },
            {"$limit": top_k},
            {"$project": {"id": 1, "_id": 0}}
        ]

    def _get_semantic_pipeline(self, query: str, top_k: int) -> List[dict]:
        """Get semantic search pipeline definition (without embedding vector)."""
        return [
            {
                "$vectorSearch": {
                    "index": "default_vector",
                    "path": "embeddings",
                    "queryVector": "<embedding_vector>",
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {"$project": {"id": 1, "_id": 0}}
        ]

    def _bm25_search(self, query: str, top_k: int) -> List[int]:
        """BM25 text search on title field. Results are sorted by relevance (highest first)."""
        logger.info(f"Running BM25 search for query: '{query}'")

        pipeline = self._get_bm25_pipeline(query, top_k)
        results = list(self.collection.aggregate(pipeline))
        doc_ids: List[int] = [doc['id'] for doc in results]
        logger.info(f"BM25 search returned {len(doc_ids)} results")
        return doc_ids

    def _semantic_search(self, query: str, top_k: int) -> List[int]:
        """Vector/semantic search using embeddings. Results are sorted by similarity (highest first)."""

        # Generate query embedding
        query_vector = self.embedding_service.generate_embedding(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default_vector",
                    "path": "embeddings",
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k
                }
            },
            {"$project": {"id": 1, "_id": 0}}
        ]

        results = list(self.collection.aggregate(pipeline))
        doc_ids: List[int] = [doc['id'] for doc in results]
        logger.info(f"Semantic search returned {len(doc_ids)} results")
        return doc_ids

    def _deduplicate_candidates(self, doc_ids: List[int]) -> Set[int]:
        """Deduplicate document IDs."""
        logger.info(f"Deduplicating {len(doc_ids)} document IDs")
        unique_ids: Set[int] = set(doc_ids)
        logger.info(f"Deduplicated to {len(unique_ids)} unique IDs")
        return unique_ids

    def _fetch_documents(self, doc_ids: Set[int]) -> List[dict]:
        """Fetch full documents for given IDs."""
        results = list(self.collection.find(
            {"id": {"$in": list(doc_ids)}},
            {"_id": 0, "id": 1, "title": 1, "score": 1, "url": 1, "time": 1}
        ))

        return results
