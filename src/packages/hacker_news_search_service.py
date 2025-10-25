import logging
from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field, field_serializer
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HackerNewsStory(BaseModel):
    """Model for Hacker News story."""
    id: int = Field(description="Hacker News story ID")
    score: float = Field(description="Story vote score")
    title: str = Field(description="Story title")
    time: datetime = Field(description="Story submission time")
    url: str = Field(description="Story URL")

    @field_serializer('time', when_used='json')
    def serialize_time(self, value: datetime) -> str:
        """Serialize datetime to RFC3339/ISO 8601 format with timezone for JSON output."""
        # Ensure timezone info is present (assume UTC if naive)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class HackerNewsSearchService:
    """Service for searching Hacker News stories in MongoDB."""

    def __init__(self, mongo_client: MongoClient, database_name: str, collection_name: str):
        """Initialize Hacker News search service."""
        self.mongo_client = mongo_client
        self.database_name = database_name
        self.collection_name = collection_name

    def search(self, query: str) -> List[HackerNewsStory]:
        """Search for Hacker News stories by query string."""
        try:
            db = self.mongo_client[self.database_name]
            collection = db[self.collection_name]

            pipeline = [
                {
                    "$search": {
                        "index": "default_text",
                        "text": {
                            "query": query,
                            "path": "title"
                        }
                    }
                },
                {
                    "$project": {
                        "id": 1,
                        "score": 1,
                        "title": 1,
                        "time": 1,
                        "url": 1,
                    }
                }
            ]
            results = list(collection.aggregate(pipeline))

            # Convert MongoDB documents to Pydantic models
            return [HackerNewsStory(**doc) for doc in results]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
