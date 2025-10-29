"""
MCP tool for searching Hacker News stories.
"""

from typing import List

from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from src.packages.search_service import HackerNewsSearchService, HackerNewsStory

SEARCH_TOOL_NAME = 'search'


class SearchResults(BaseModel):
    """Container for search results."""
    results: List[HackerNewsStory] = Field(description="List of matching Hacker News stories")
    total: int = Field(description="Total number of results returned")


class HackerNewsSearchTool():
    def __init__(self, hacker_news_search_service: HackerNewsSearchService):
        self.name = SEARCH_TOOL_NAME
        self.title = 'Search Hacker News stories'
        self.description = 'Search Hacker News stories using hybrid text and vector search.'
        self.annotations = ToolAnnotations(title="Hacker News Search Tool")
        self.structured_output = True
        self.hacker_news_search_service = hacker_news_search_service

    def execute(
        self,
        query: str = Field(
            description="The search query to find matching Hacker News stories. Supports natural language queries."),
        limit: int = Field(
            default=10,
            description="Maximum number of results to return. Default is 10.",
            ge=1,
            le=100)
    ) -> SearchResults:
        """Search for Hacker News stories using hybrid text and vector search."""
        stories = self.hacker_news_search_service.search(query, limit)
        return SearchResults(results=stories, total=len(stories))
