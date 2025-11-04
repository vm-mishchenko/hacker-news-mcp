"""
Search for Hacker News stories using hybrid text + vector search.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer
from pymongo import MongoClient

# Configure logging
logger = logging.getLogger(__name__)


class HackerNewsStory(BaseModel):
    """Model for Hacker News story."""
    id: int = Field(description="Hacker News story ID")
    score: float = Field(description="Story vote score")
    title: str = Field(description="Story title")
    time: datetime = Field(description="Story submission time")
    url: Optional[str] = Field(default=None, description="Story URL")

    @field_serializer('time', when_used='json')
    def serialize_time(self, value: datetime) -> str:
        """Serialize datetime to RFC3339/ISO 8601 format with timezone for JSON output."""
        # Ensure timezone info is present (assume UTC if naive)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class HackerNewsSearchService:
    """Service for searching Hacker News stories in MongoDB."""

    def __init__(
        self,
        mongo_client: MongoClient,
        database_name: str,
        collection_name: str,
        embedding_service
    ):
        """Initialize Hacker News search service."""
        self.mongo_client = mongo_client
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_service = embedding_service

    def search(
        self,
        query: Annotated[str, Field(
            description="The search query to find matching Hacker News stories. Supports natural language queries.")],
        limit: Annotated[int, Field(default=10,
                                    description="Maximum number of results to return. Default is 10.",
                                    ge=1, le=100)],
        from_time: Annotated[Optional[datetime], Field(
            default=None,
            description="Start of the time window for filtering stories. Only stories submitted on or after this time will be included. Example: '2024-01-01T00:00:00Z'")] = None,
        to_time: Annotated[Optional[datetime], Field(
            default=None,
            description="End of the time window for filtering stories. Only stories submitted on or before this time will be included. Example: '2024-12-31T23:59:59Z'")] = None
    ) -> List[HackerNewsStory]:
        """Search for Hacker News stories by query string. Uses vector search to boost text search results."""
        logger.info(
            f"Starting search for query: '{query}' with limit: {limit}, from_time: {from_time}, to_time: {to_time}")

        # Enforce limit range
        if limit < 1 or limit > 100:
            logger.error(f"Invalid limit: {limit}. Must be between 1 and 100.")
            raise ValueError(f"Limit must be between 1 and 100, got {limit}")

        try:
            # Run vector search to get relevant document IDs
            vector_results = self._run_vector_search(query, limit, from_time, to_time)
            if len(vector_results) == 0:
                logger.warning(f"Vector search returned 0 results for query: '{query}'")
            else:
                logger.info(f"Vector search returned {len(vector_results)} results")

            # Run text search with boosting for vector search results
            text_results = self._run_text_search(query, limit, from_time, to_time)
            if len(text_results) == 0:
                logger.warning(f"Text search returned 0 results for query: '{query}'")
            else:
                logger.info(f"Text search returned {len(text_results)} results")

            # Fuse results using Reciprocal Rank Fusion
            fused_results = self._fusion_search_results(vector_results, text_results, limit)
            logger.info(f"Fusion returned {len(fused_results)} results")

            return fused_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _fusion_search_results(
        self,
        search_results_a: List[HackerNewsStory],
        search_results_b: List[HackerNewsStory],
        limit: int,
        k: int = 60
    ) -> List[HackerNewsStory]:
        """Fuse two search result lists using Reciprocal Rank Fusion (RRF)."""
        logger.info(f"Starting RRF fusion with k={k}, limit={limit}")

        # Calculate RRF scores for all documents
        rrf_scores: Dict[int, float] = self._calculate_rrf_scores(
            search_results_a, search_results_b, k
        )
        logger.info(f"Calculated RRF scores for {len(rrf_scores)} unique documents")

        # Create a map of document ID to story for quick lookup
        doc_map: Dict[int, HackerNewsStory] = {}
        for story in search_results_a + search_results_b:
            if story.id not in doc_map:
                doc_map[story.id] = story

        # Sort documents by RRF score (descending) and take top limit
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id],
                                reverse=True)
        top_doc_ids = sorted_doc_ids[:limit]

        # Build final result list
        fused_results: List[HackerNewsStory] = [doc_map[doc_id] for doc_id in top_doc_ids]

        logger.info(f"RRF fusion complete: returning {len(fused_results)} results")
        return fused_results

    def _calculate_rrf_scores(
        self,
        search_results_a: List[HackerNewsStory],
        search_results_b: List[HackerNewsStory],
        k: int
    ) -> Dict[int, float]:
        """Calculate RRF scores for documents from two ranked lists."""
        logger.info("Calculating RRF scores")

        # Map of document ID (int) to RRF score (float)
        rrf_scores: Dict[int, float] = {}

        # Process first result list
        for rank, story in enumerate(search_results_a, start=1):
            rrf_scores[story.id] = 1.0 / (k + rank)

        # Process second result list
        for rank, story in enumerate(search_results_b, start=1):
            if story.id in rrf_scores:
                rrf_scores[story.id] += 1.0 / (k + rank)
            else:
                rrf_scores[story.id] = 1.0 / (k + rank)

        logger.info(f"RRF scores calculated for {len(rrf_scores)} documents")
        return rrf_scores

    def _run_vector_search(
        self,
        query: str,
        limit: int,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[HackerNewsStory]:
        """Execute vector search using embeddings. Returns only document IDs."""
        logger.info(f"Running vector search for query: '{query}' and limit {limit}")

        # Generate embedding for query
        query_vector = self.embedding_service.generate_embedding(query)

        db = self.mongo_client[self.database_name]
        collection = db[self.collection_name]

        # Calculate numCandidates (typically 10-20x the limit)
        num_candidates = limit * 10

        # Build vector search stage
        vector_search_stage = {
            "$vectorSearch": {
                "index": "default_vector",
                "path": "embeddings",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        }

        # Add time filter if specified
        if from_time is not None or to_time is not None:
            time_filter: Dict[str, Any] = {}
            if from_time is not None:
                time_filter["$gte"] = from_time
            if to_time is not None:
                time_filter["$lte"] = to_time
            vector_search_stage["$vectorSearch"]["filter"] = {"time": time_filter}

        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "score": 1,
                    "title": 1,
                    "time": 1,
                    "url": 1
                }
            }
        ]

        results = list(collection.aggregate(pipeline))
        return [HackerNewsStory(**doc) for doc in results]

    def _run_text_search(
        self,
        query: str,
        limit: int,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[HackerNewsStory]:
        """Execute text search with boosting for specific document IDs, story score, and recency.

        Final query structure:
        {
            "compound": {
                "must": [
                    {
                        "text": {
                            "query": "...",
                            "path": "title",
                            "score": {"boost": {"path": "score", "undefined": 1}}
                        }
                    }
                ],
                "should": [
                    {"near": {"path": "time", "origin": <now>, "pivot": 86400000, "score": {"boost": {"value": 1.5}}}},
                ],
                "filter": [
                    {"range": {"path": "time", "gte": <from_time>, "lte": <to_time>}}
                ]
            }
        }

        The "must" clause ensures all results match the text query and boosts by HN vote score.
        The "should" clauses boost documents by recency and from vector search with decreasing weights.
        The "filter" clause applies time window constraints if specified.
        """
        logger.info(f"Running text search for query: '{query}")
        db = self.mongo_client[self.database_name]
        collection = db[self.collection_name]

        # Build compound query with text search and score boosting
        compound_clauses = {
            "must": [
                {
                    "text": {
                        "query": query,
                        "path": "title",
                        "fuzzy": {
                            "prefixLength": 1,
                            "maxEdits": 2,
                            "maxExpansions": 100
                        },
                        "score": {
                            "boost": {
                                "path": "score",
                                "undefined": 1
                            }
                        }
                    }
                }
            ],
            "should": [
                {
                    "near": {
                        "path": "time",
                        "origin": datetime.now(timezone.utc),
                        "pivot": 86400000  # 1 day in milliseconds - natural decay: older posts get lower scores
                    }
                }
            ]
        }

        # Add time filter if specified
        if from_time is not None or to_time is not None:
            filter_clauses: List[Dict[str, Any]] = []
            range_filter: Dict[str, Any] = {"path": "time"}
            if from_time is not None:
                range_filter["gte"] = from_time
            if to_time is not None:
                range_filter["lte"] = to_time
            filter_clauses.append({"range": range_filter})
            compound_clauses["filter"] = filter_clauses

        pipeline = [
            {
                "$search": {
                    "index": "default_text",
                    "compound": compound_clauses
                }
            },
            {"$limit": limit},
            {
                "$project": {
                    "_id": 0,
                    "id": 1,
                    "score": 1,
                    "title": 1,
                    "time": 1,
                    "url": 1
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        # Convert to Pydantic models
        return [HackerNewsStory(**doc) for doc in results]

    def _extract_doc_ids(self, vector_results: List[dict]) -> List[int]:
        """Extract document IDs from vector search results."""
        logger.info(f"Extracting document IDs from {len(vector_results)} vector results")
        doc_ids: List[int] = [doc['id'] for doc in vector_results]
        logger.info(f"Extracted {len(doc_ids)} document IDs")
        return doc_ids
