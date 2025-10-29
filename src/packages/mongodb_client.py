"""
MongoDB client factory for creating database connections.
"""

import logging
from urllib.parse import quote_plus

from pymongo import MongoClient

# Configure logging
logger = logging.getLogger(__name__)


class MongoDBClient:
    """Factory for creating MongoDB client connections."""

    def __init__(self, username: str, password: str, uri: str):
        """Initialize MongoDB client configuration."""
        self.username = username
        self.password = password
        self.uri = uri

    def build_connection_string(self, base_uri: str, username: str, password: str) -> str:
        """
        Construct MongoDB URI with credentials.
        Expected format: mongodb+srv://cluster.mongodb.net/?retryWrites=true&w=majority&appName=xxx
        """
        # Parse the URI to inject credentials
        if "://" not in base_uri:
            raise ValueError(
                "Invalid MongoDB URI format. Expected format: mongodb+srv://cluster.mongodb.net/?retryWrites=true&w=majority")

        protocol, rest = base_uri.split("://", 1)

        # Remove any existing credentials (username:password@)
        if "@" in rest:
            rest = rest.split("@", 1)[1]

        # URL-encode credentials to handle special characters
        encoded_username = quote_plus(username)
        encoded_password = quote_plus(password)

        # Construct URI with credentials
        # Format: mongodb+srv://username:password@cluster.mongodb.net/?params
        mongodb_uri = f"{protocol}://{encoded_username}:{encoded_password}@{rest}"

        return mongodb_uri

    def get_client(self) -> MongoClient:
        """Create and return a new MongoDB client instance."""
        connection_string = self.build_connection_string(self.uri, self.username, self.password)
        return MongoClient(connection_string)
