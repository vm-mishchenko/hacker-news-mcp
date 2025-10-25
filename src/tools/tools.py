from typing import List

from mcp.types import ToolAnnotations
from pydantic import Field

from packages.hacker_news_search_service import HackerNewsSearchService, HackerNewsStory


class HackerNewsSearchTool():
    def __init__(self, hacker_news_search_service: HackerNewsSearchService):
        self.name = HackerNewsSearchTool.__name__
        self.title = 'Search Hacker News Stories'
        self.description = 'Search for Hacker News stories by query string'
        self.annotations = ToolAnnotations(title="Hacker News Search Tool")
        self.structured_output = True
        self.hacker_news_search_service = hacker_news_search_service

    def execute(self,
                query: str = Field(
                    description="The search query to find matching Hacker News stories.")) -> List[
        HackerNewsStory]:
        """Search for Hacker News stories by query string."""
        return self.hacker_news_search_service.search(query)
