"""
Generate search queries from popular Hacker News stories using LLM.

This script:
1. Fetches top N stories from MongoDB (sorted by recency, then score)
2. Generates realistic search queries for each story using OpenAI
3. Outputs stories with their queries in JSON format
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from src.evaluation.packages.query_generation_service import OpenAIQueryGenerationService
from src.packages.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

# Load .env.local from project root (must run from project root)
env_local_path = Path('.env.local')
if env_local_path.exists():
    load_dotenv(env_local_path)
    logger.info("Loaded .env.local for local development")
else:
    logger.info("No .env.local file found")


@dataclass
class HackerNewsStory:
    """Represents a Hacker News story."""
    id: str
    title: str
    score: int


@dataclass
class StoryWithQueries:
    """Represents a story with its generated queries."""
    title: str
    queries: List[str]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    logger.info("Parsing CLI arguments")

    parser = argparse.ArgumentParser(
        description="Generate search queries from popular Hacker News stories"
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=3,
        help="Number of top stories to process (default: 3)"
    )
    parser.add_argument(
        "--queries-per-story",
        type=int,
        default=5,
        help="Number of queries to generate per story (default: 5)"
    )

    args = parser.parse_args()
    logger.info(
        f"Parsed arguments: max_stories={args.max_stories}, queries_per_story={args.queries_per_story}")
    return args


def load_config() -> Dict[str, str]:
    """Load configuration from environment variables."""
    logger.info("Loading configuration from environment")

    # .env.local is already loaded at module level
    config = {
        'mongodb_username': os.environ['MONGODB_USERNAME'],
        'mongodb_password': os.environ['MONGODB_PASSWORD'],
        'mongodb_uri': os.environ['MONGODB_URI'],
        'openai_api_key': os.environ['OPENAI_API_KEY'],
    }

    logger.info("Configuration loaded successfully")
    return config


def query_mongodb(mongo_client: MongoDBClient, limit: int) -> List[Dict]:
    """Query MongoDB for top stories by score and recency."""
    logger.info(f"Querying MongoDB for top {limit} stories by score and recency")

    client = mongo_client.get_client()
    db = client['hacker-news']
    collection = db['posts']

    # Query for top stories sorted by time (most recent first), then by score
    # This prioritizes recent stories while still considering popularity
    cursor = collection.find(
        {},
        {
            "_id": 0,
            "id": 1,
            "title": 1,
            "score": 1,
            "time": 1
        }
    ).sort([("time", -1), ("score", -1)]).limit(limit)

    stories = list(cursor)
    logger.info(f"Fetched {len(stories)} stories from MongoDB")

    return stories


def fetch_stories(mongo_client: MongoDBClient, limit: int) -> List[HackerNewsStory]:
    """Fetch top stories from MongoDB."""
    logger.info(f"Starting to fetch top {limit} stories")

    raw_stories = query_mongodb(mongo_client, limit)

    # Convert to dataclass
    stories: List[HackerNewsStory] = []
    for story in raw_stories:
        stories.append(HackerNewsStory(
            id=str(story['id']),
            title=story['title'],
            score=story['score']
        ))

    logger.info(f"Converted {len(stories)} stories to HackerNewsStory objects")
    return stories


def generate_queries_for_story(
    llm_service: OpenAIQueryGenerationService,
    story: HackerNewsStory,
    num_queries: int
) -> List[str]:
    """Generate queries for a single story."""
    logger.info(f"Generating {num_queries} queries for story: {story.title}")

    queries = llm_service.generate_queries(story.title, num_queries)

    logger.info(f"Generated {len(queries)} queries for story ID {story.id}")
    return queries


def generate_all_queries(
    stories: List[HackerNewsStory],
    llm_service: OpenAIQueryGenerationService,
    queries_per_story: int
) -> tuple[List[str], List[StoryWithQueries]]:
    """Generate queries for all stories."""
    logger.info(f"Starting query generation for {len(stories)} stories")

    all_queries: List[str] = []
    stories_with_queries: List[StoryWithQueries] = []

    for idx, story in enumerate(stories, 1):
        logger.info(f"Processing story {idx}/{len(stories)}: {story.title}")

        queries = generate_queries_for_story(llm_service, story, queries_per_story)
        all_queries.extend(queries)

        # Store story with its queries
        stories_with_queries.append(StoryWithQueries(
            title=story.title,
            queries=queries
        ))

        logger.info(f"Total queries so far: {len(all_queries)}")

    logger.info(f"Generated {len(all_queries)} total queries from {len(stories)} stories")
    return all_queries, stories_with_queries


def write_jsonl(queries: List[str], output_path: str) -> None:
    """Write queries to JSONL file."""
    logger.info(f"Writing {len(queries)} queries to {output_path}")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with open(output_path, 'w') as f:
        for query in queries:
            json_line = json.dumps({"query": query})
            f.write(json_line + '\n')

    logger.info(f"Successfully wrote {len(queries)} queries to {output_path}")


def write_stories_json(stories_with_queries: List[StoryWithQueries], output_path: str) -> None:
    """Write stories with queries to JSON file."""
    logger.info(f"Writing {len(stories_with_queries)} stories with queries to {output_path}")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    stories_data = [
        {
            "title": story.title,
            "queries": story.queries
        }
        for story in stories_with_queries
    ]

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(stories_data, f, indent=2)

    logger.info(f"Successfully wrote {len(stories_with_queries)} stories to {output_path}")


def write_output(queries: List[str], stories_with_queries: List[StoryWithQueries]) -> str:
    """Write queries to output files."""
    logger.info(f"Starting output write")

    # Create output file with timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_path = f"evaluation/candidate_search_queries_{timestamp}.json"
    logger.info(f"Output will be written to: {output_path}")

    # Write JSON file with stories and their queries
    write_stories_json(stories_with_queries, output_path)

    logger.info("Output write complete")
    return output_path


def run() -> None:
    """Main coordinator function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting query generation script")

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config()

    # Initialize MongoDB client
    logger.info("Initializing MongoDB client")
    mongo_client = MongoDBClient(
        username=config['mongodb_username'],
        password=config['mongodb_password'],
        uri=config['mongodb_uri']
    )
    logger.info("MongoDB client initialized")

    # Initialize LLM service
    logger.info("Initializing OpenAI query generation service")
    llm_service = OpenAIQueryGenerationService(
        api_key=config['openai_api_key'],
        model="gpt-4o"
    )
    logger.info("OpenAI query generation service initialized")

    # Fetch stories
    stories = fetch_stories(mongo_client, args.max_stories)

    # Generate queries
    all_queries, stories_with_queries = generate_all_queries(stories, llm_service, args.queries_per_story)

    # Write output
    output_path = write_output(all_queries, stories_with_queries)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Stories processed: {len(stories)}")
    logger.info(f"Queries generated: {len(all_queries)}")
    logger.info(f"Output file: {output_path}")
    logger.info("=" * 80)
    logger.info("Query generation script complete")


if __name__ == "__main__":
    run()
