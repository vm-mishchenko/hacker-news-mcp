"""
MPC Retriever implementation for evaluation framework.

Connects to the running MCP server and retrieves document IDs using the HackerNewsSearchTool.
"""

import asyncio
import json
import logging
from typing import List

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.packages.evaluation_framework import Retriever, QueryText, DocumentId
from src.tools.tools import SearchResults, SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)

# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class MPCRetriever(Retriever):
    """Retriever that connects to MCP server to get search results."""

    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        """Initialize MPC retriever with server URL."""
        logger.info(f"Initializing MPCRetriever with server_url={server_url}")
        self.server_url = server_url
        self._session = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        logger.info("MPCRetriever initialized successfully")

    async def _connect(self):
        """Establish connection to MCP server."""
        logger.info(f"Connecting to MCP server at {self.server_url}")

        self._client_context = streamablehttp_client(self.server_url)
        self._read_stream, self._write_stream, _ = await self._client_context.__aenter__()

        self._session_context = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_context.__aenter__()

        await self._session.initialize()
        logger.debug("MCP session established")

    def retrieve(self, query: QueryText, k: int = 20) -> List[DocumentId]:
        """Retrieve documents for a query by calling the MCP server."""
        logger.debug(f"Starting retrieval for query: '{query}' with k={k}")

        search_results = self._loop.run_until_complete(self._search(query, k))
        logger.debug(f"Received {search_results.total} search results from MCP server")

        doc_ids = self._extract_doc_ids(search_results)
        logger.debug(f"Extracted {len(doc_ids)} document IDs")

        return doc_ids

    async def _search(self, query: QueryText, limit: int) -> SearchResults:
        """Execute search using the persistent session."""
        result = await self._session.call_tool(
            SEARCH_TOOL_NAME,
            arguments={"query": query, "limit": limit}
        )
        logger.debug("Search tool called successfully")

        search_results = self._parse_tool_response(result.content)
        logger.debug(f"Parsed SearchResults with {search_results.total} results")

        return search_results

    def _parse_tool_response(self, content: List) -> SearchResults:
        """Parse MCP tool response content and deserialize to SearchResults object."""
        logger.debug(f"Parsing tool response content, content length: {len(content)}")

        if not content or len(content) == 0:
            logger.error("Empty content received from tool")
            raise ValueError("Empty content received from MCP tool")

        # For structured output, the content contains a SearchResults object as JSON
        if len(content) > 0 and hasattr(content[0], 'text'):
            try:
                # Parse JSON from text content
                parsed_json = json.loads(content[0].text)

                # Deserialize to SearchResults Pydantic object
                search_results = SearchResults(**parsed_json)
                logger.debug(
                    f"Successfully deserialized SearchResults with {search_results.total} results")
                return search_results
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from text content: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to deserialize SearchResults: {e}")
                raise
        else:
            error_msg = f"Unexpected content item type: {type(content[0]) if content else 'empty'}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _extract_doc_ids(self, search_results: SearchResults) -> List[DocumentId]:
        """Extract document IDs from SearchResults object."""
        logger.debug(
            f"Extracting document IDs from SearchResults with {len(search_results.results)} results")

        doc_ids: List[DocumentId] = []

        for story in search_results.results:
            # Convert story ID to string and create DocumentId
            doc_id = DocumentId(str(story.id))
            doc_ids.append(doc_id)

        logger.debug(f"Extracted {len(doc_ids)} document IDs")
        return doc_ids
