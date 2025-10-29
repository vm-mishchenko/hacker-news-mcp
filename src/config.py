"""
Configuration management for MCP server settings and command-line arguments.
"""

import argparse
from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Transport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class Config(BaseSettings):
    host: str = Field("0.0.0.0", alias="MCP_HOST", description="Host to bind")
    port: int = Field(8000, alias="MCP_PORT", description="Port to listen on")
    transport: Transport = Field(
        Transport.STREAMABLE_HTTP,
        description=f"Transport protocol, allowed: {[t.value for t in Transport]}"
    )
    log_level: str = Field('INFO', description="Logging level",
                           examples=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    MONGODB_DATABASE_NAME: str = Field(default="hacker-news", description="MongoDB database name")
    MONGODB_COLLECTION_NAME: str = Field(default="posts", description="MongoDB collection name")
    MONGODB_USERNAME: str = Field(description="Mongodb user")
    MONGODB_PASSWORD: str = Field(description="Mongodb user")
    MONGODB_URI: str = Field(
        description="Mongodb uri. Example: mongodb+srv://@hacker-news-assistant.wglor0l.mongodb.net/?appName=hacker-news-assistant")
    OPENAI_API_KEY: str = Field(description="OpenAI API key for embedding generation")

    @field_validator("transport")
    def reject_sse(cls, v):
        if v == Transport.SSE:
            raise ValueError("SSE transport not supported")
        return v


def parse_args() -> argparse.Namespace:
    """Parse command line arguments and environment variables into Config."""
    parser = argparse.ArgumentParser(description="MCP Playground Server")

    # Transport options
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        help="Transport protocol to use",
    )

    # HTTP transport configuration
    parser.add_argument(
        "--host",
        help="Host to bind to for HTTP transports (default: 0.0.0.0, env: MCP_HOST)",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on for HTTP transports",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        help="Enable debug logging",
    )

    # Optional log level
    parser.add_argument(
        "--log-level",
        help="Logging level",
    )

    args = parser.parse_args()

    return args


def get_config():
    args = parse_args()
    # Only include CLI values that are actually set
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**cli_overrides)
