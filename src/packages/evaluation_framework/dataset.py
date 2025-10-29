"""
Dataset management for ground truth queries and relevance labels.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path

from .models import QueryResult, QueryText, DocumentId

logger = logging.getLogger(__name__)


class Dataset:
    """Ground truth: queries + relevance labels."""

    def __init__(self, queries: List[QueryResult]):
        """Initialize dataset with queries."""
        logger.info(f"Initializing dataset with {len(queries)} queries")
        self._queries = queries
        self._query_map: Dict[str, QueryResult] = {q.query: q for q in queries}
        self._validate()
        logger.info("Dataset initialized successfully")

    def _validate(self) -> None:
        """Validate dataset integrity."""
        logger.info("Validating dataset")

        # Check for duplicate queries
        query_texts = [q.query for q in self._queries]
        if len(query_texts) != len(set(query_texts)):
            # Find and report duplicates
            seen = set()
            duplicates = []
            for query_text in query_texts:
                if query_text in seen:
                    duplicates.append(query_text)
                seen.add(query_text)

            duplicate_list = "\n".join(f"  - {q}" for q in duplicates)
            raise ValueError(f"Duplicate queries found in dataset:\n{duplicate_list}")

        # Validate each query
        for query in self._queries:
            # Warn if query has no relevant docs
            if len(query.relevant_docs) == 0:
                logger.warning(f"Query '{query.query}' has 0 relevant documents")

        logger.info("Dataset validation complete")

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        """Load dataset from JSONL file."""
        logger.info(f"Loading dataset from {path}")

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        queries: List[QueryResult] = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)

                    # Parse relevant document IDs
                    relevant_docs = [
                        DocumentId(doc_id)
                        for doc_id in data.get('relevant_docs', [])
                    ]

                    # Create query
                    query = QueryResult(
                        query=QueryText(data['query']),
                        relevant_docs=relevant_docs
                    )

                    queries.append(query)

                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(
                        f"Error parsing line {line_num} in {path}: {e}"
                    ) from e
        
        logger.info(f"Loaded {len(queries)} queries from {path}")
        return cls(queries)

    def get_queries(self) -> List[QueryResult]:
        """Get all queries."""
        return self._queries

    def get_query(self, query_text: str) -> QueryResult:
        """Get specific query by query text."""
        if query_text not in self._query_map:
            raise KeyError(f"Query not found: {query_text}")
        return self._query_map[query_text]

