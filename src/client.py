"""
Example client for testing the Hacker News MCP server.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.tools.tools import SEARCH_TOOL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MCP = "http://localhost:8000/mcp"
PROD_MCP = "https://hacker-news-mcp-632359460921.us-central1.run.app/mcp"


async def main():
    # Connect to the server using Streamable HTTP
    async with streamablehttp_client(LOCAL_MCP) as (
            read_stream,
            write_stream,
            get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            logger.info("Available tools:")
            for tool in tools_result.tools:
                logger.info(f"  - {tool.name}: {tool.description}")

            # Call Hacker News search tool
            query = "agentic"
            from_time = (datetime.now(timezone.utc) - timedelta(days=360)).isoformat()
            print(from_time)
            result = await session.call_tool(SEARCH_TOOL_NAME,
                                             arguments={
                                                 "query": query,
                                                 "limit": 10,
                                                 "from_time": from_time
                                             })

            # Format and log results in human-readable format
            logger.info(f"Search: {query}\n")
            if not result.content:
                logger.info("No results found.")
            else:
                # Parse the response - now it's a single SearchResults object
                response_data = json.loads(result.content[0].text)
                stories = response_data.get('results', [])
                total = response_data.get('total', 0)

                logger.info(f"Found {total} results:\n")
                for idx, story in enumerate(stories):
                    formatted_time = datetime.fromisoformat(story['time']).strftime('%Y-%m-%d')
                    logger.info(
                        f"{idx}. {story['title']}, ID: {story['id']}, Score: {story['score']}, Time: {formatted_time}")


if __name__ == "__main__":
    asyncio.run(main())
