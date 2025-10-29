"""
Abstract retriever interface for evaluation framework.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from .models import DocumentId, QueryText

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Abstract interface for any retrieval system you test."""

    @abstractmethod
    def retrieve(self, query: QueryText, k: int = 10) -> List[DocumentId]:
        """Retrieve ranked list of document IDs for a query."""
        pass
