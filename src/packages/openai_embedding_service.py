"""
OpenAI embedding service for generating text embeddings.

Functions:
- generate_embedding (executor) - Generate embedding for a single text query
"""

import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    """Service for generating embeddings using OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding service.

        Args:
            api_key: OpenAI API key
            model: Embedding model name (default: text-embedding-3-small)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI embedding service with model: {model}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for input text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            Exception: If OpenAI API call fails
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=512
            )
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
