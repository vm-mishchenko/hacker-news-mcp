"""
Search for Hacker News stories using hybrid text + vector search.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, List, Optional

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
                                    ge=1, le=100)]
    ) -> List[HackerNewsStory]:
        """Search for Hacker News stories by query string. Uses vector search to boost text search results."""
        logger.info(f"Starting search for query: '{query}' with limit: {limit}")

        # Enforce limit range
        if limit < 1 or limit > 100:
            logger.error(f"Invalid limit: {limit}. Must be between 1 and 100.")
            raise ValueError(f"Limit must be between 1 and 100, got {limit}")

        try:
            # Run vector search to get relevant document IDs
            vector_results = self._run_vector_search(query, limit)
            logger.info(f"Vector search returned {len(vector_results)} results")
            return vector_results

            # Extract document IDs from vector search results
            boost_doc_ids = self._extract_doc_ids(vector_results)
            logger.info(f"Extracted {len(boost_doc_ids)} document IDs for boosting")

            # Run text search with boosting for vector search results
            boosted_results = self._run_text_search(query, boost_doc_ids, limit)
            logger.info(f"Boosted text search returned {len(boosted_results)} results")

            return boosted_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _run_vector_search(self, query: str, limit: int) -> List[HackerNewsStory]:
        """Execute vector search using embeddings. Returns only document IDs."""
        logger.info(f"Running vector search for query: '{query}' and limit {limit}")

        # Generate embedding for query
        query_vector = self.embedding_service.generate_embedding(query)

        db = self.mongo_client[self.database_name]
        collection = db[self.collection_name]

        # Calculate numCandidates (typically 10-20x the limit)
        num_candidates = limit * 10

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "default_vector",
                    "path": "embeddings",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            },
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
        logger.info(f"Vector search returned {len(results)} results")
        return [HackerNewsStory(**doc) for doc in results]

    def _run_text_search(self, query: str, boost_doc_ids: List[int], limit: int) -> List[
        HackerNewsStory]:
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
                    {"equals": {"path": "id", "value": 123, "score": {"boost": {"value": 2.0}}}},
                    {"equals": {"path": "id", "value": 456, "score": {"boost": {"value": 1.9}}}},
                    {"equals": {"path": "id", "value": 789, "score": {"boost": {"value": 1.8}}}},
                    ...
                ]
            }
        }

        The "must" clause ensures all results match the text query and boosts by HN vote score.
        The "should" clauses boost documents by recency and from vector search with decreasing weights.
        """
        logger.info(
            f"Running text search for query: '{query}' with {len(boost_doc_ids)} boost IDs")
        db = self.mongo_client[self.database_name]
        collection = db[self.collection_name]

        # Build compound query with text search and score boosting
        compound_clauses = {
            "must": [
                {
                    "text": {
                        "query": query,
                        "path": "title",
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
                        "pivot": 86400000,  # 1 day in milliseconds
                        "score": {"boost": {"value": 1.5}}
                    }
                }
            ]
        }

        # Add should clauses for boosting vector search results
        # Each subsequent result gets a slightly lower boost to reflect vector search ranking
        if boost_doc_ids:
            compound_clauses["should"].extend([
                {
                    "equals": {
                        "path": "id",
                        "value": doc_id,
                        "score": {"boost": {"value": max(1.0, 2.0 - (index * 0.1))}}
                    }
                }
                for index, doc_id in enumerate(boost_doc_ids)
            ])

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
        logger.info(f"Boosted text search returned {len(results)} results")

        # Convert to Pydantic models
        return [HackerNewsStory(**doc) for doc in results]

    def _extract_doc_ids(self, vector_results: List[dict]) -> List[int]:
        """Extract document IDs from vector search results."""
        logger.info(f"Extracting document IDs from {len(vector_results)} vector results")
        doc_ids: List[int] = [doc['id'] for doc in vector_results]
        logger.info(f"Extracted {len(doc_ids)} document IDs")
        return doc_ids
