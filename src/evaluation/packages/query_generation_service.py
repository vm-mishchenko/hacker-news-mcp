"""
OpenAI query generation service for generating search queries from story titles.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

# Load prompt template from file
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "prompts" / "generate_queries.md"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()


class OpenAIQueryGenerationService:
    """Service for generating search queries using OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize OpenAI query generation service."""
        logger.info(f"Initializing OpenAI query generation service with model: {model}")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI query generation service with model: {model}")

    def _build_prompt(self, title: str, num_queries: int) -> str:
        """Build the prompt for query generation."""
        logger.info(f"Building prompt for title: {title}, num_queries: {num_queries}")

        prompt = PROMPT_TEMPLATE.format(title=title, num_queries=num_queries)

        logger.info("Prompt built successfully")
        return prompt

    def generate_queries(self, title: str, num_queries: int = 5, max_retries: int = 3) -> List[str]:
        """Generate search queries for a given story title."""
        logger.info(f"Starting query generation for title: {title}")
        logger.info(f"Generating {num_queries} queries with max {max_retries} retries")

        prompt = self._build_prompt(title, num_queries)

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} to generate queries")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                logger.info(f"Received response from OpenAI API")

                # Parse JSON response
                try:
                    # Try to parse as direct array
                    queries = json.loads(content)

                    # Handle both array and object with array field
                    if isinstance(queries, dict):
                        # Look for common field names
                        for key in ['queries', 'results', 'data', 'items']:
                            if key in queries and isinstance(queries[key], list):
                                queries = queries[key]
                                break
                        else:
                            # If no known field, take first list value
                            for value in queries.values():
                                if isinstance(value, list):
                                    queries = value
                                    break
                            else:
                                # If dict has string values (query -> category mapping), extract keys
                                if all(isinstance(v, str) for v in queries.values()):
                                    logger.info(
                                        "Detected query-to-category mapping, extracting keys")
                                    queries = list(queries.keys())

                    if not isinstance(queries, list):
                        raise ValueError(f"Expected list, got {type(queries)}")

                    # Validate all items are strings and clean whitespace
                    cleaned_queries: List[str] = []
                    for q in queries:
                        if not q:
                            continue

                        # Convert to string and strip whitespace
                        query_str = str(q).strip()

                        # Skip empty queries after stripping
                        if not query_str:
                            logger.warning(f"Skipping empty query after stripping: '{q}'")
                            continue

                        # Normalize multiple spaces to single space
                        query_str = re.sub(r'\s+', ' ', query_str)

                        cleaned_queries.append(query_str)

                    if len(cleaned_queries) < len(queries):
                        logger.warning(
                            f"Filtered out {len(queries) - len(cleaned_queries)} queries with whitespace issues")

                    logger.info(f"Successfully generated {len(cleaned_queries)} queries")
                    return cleaned_queries

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    logger.warning(f"Raw response: {content}")

                    if attempt < max_retries - 1:
                        logger.info("Retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        logger.error("All parsing attempts failed")
                        return []

            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    return []

        logger.error("Failed to generate queries after all retries")
        return []
