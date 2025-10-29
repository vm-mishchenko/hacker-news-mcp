"""
MCP server for Hacker News search functionality.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.config import get_config, Transport
from src.packages.embedding_service import OpenAIEmbeddingService
from src.packages.mongodb_client import MongoDBClient
from src.packages.search_service import HackerNewsSearchService
from src.tools.tools import HackerNewsSearchTool

# Configure logging
logger = logging.getLogger(__name__)

# Load .env.local from project root (must run from project root)
env_local_path = Path('.env.local')
if env_local_path.exists():
    load_dotenv(env_local_path)
    logger.info("Loaded .env.local for local development")
else:
    logger.info("No .env.local file found")


def main():
    # Load configuration from environment variables and command-line arguments
    config = get_config()

    # Configure logging
    logging.basicConfig(level=config.log_level)

    # Create MCP server with configuration
    mcp = FastMCP(host=config.host, port=config.port)

    # Create MongoDB client with hardcoded credentials
    mongodb_client_factory = MongoDBClient(
        username=config.MONGODB_USERNAME,
        password=config.MONGODB_PASSWORD,
        uri=config.MONGODB_URI,
    )
    mongo_client = mongodb_client_factory.get_client()

    # Create OpenAI embedding service
    openai_embedding_service = OpenAIEmbeddingService(
        api_key=config.OPENAI_API_KEY
    )

    # Create Hacker News search service
    hacker_news_search_service = HackerNewsSearchService(
        mongo_client=mongo_client,
        database_name=config.MONGODB_DATABASE_NAME,
        collection_name=config.MONGODB_COLLECTION_NAME,
        embedding_service=openai_embedding_service
    )

    # Register tools
    tools = [
        HackerNewsSearchTool(hacker_news_search_service)
    ]

    for tool in tools:
        mcp.add_tool(tool.execute,
                     name=tool.name,
                     title=tool.title,
                     description=tool.description,
                     annotations=tool.annotations,
                     structured_output=getattr(tool, 'structured_output', None))

    # Run server with configured transport
    if config.transport == Transport.STDIO:
        logger.info("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif config.transport == Transport.STREAMABLE_HTTP:
        logger.info(
            f"Running server with Streamable HTTP transport, address http://{config.host}:{config.port}/mcp.")
        mcp.run(transport="streamable-http")
    else:
        logger.error(f"Unexpected transport: {config.transport}")
        raise ValueError(f"Unknown transport: {config.transport}")


if __name__ == "__main__":
    main()
