# Evaluation

The evaluation framework compares search results against a relevance dataset. Generate a dataset, evaluate, tune, and repeat.

## Overview

The evaluation process consists of two main components:

**Relevance Dataset Generation**
Create pairs of search queries and their most relevant story IDs. This dataset serves as the ground truth for measuring search quality.

**MCP Search Evaluation**
Run actual MCP searches using queries from the relevance dataset and compare results using standard information retrieval metrics.

**High-level flow**
```bash
# Generate search queries (output to /evaluation/)
python -m src.evaluation.generate.generate_queries

# Add queries to src/evaluation/datasets/search_queries.jsonl

# Generate relevance dataset based on the search_queries.jsonl
python -m src.evaluation.generate.generate_relevance_dataset

# Promote generated dataset to src/evaluation/datasets/relevance_dataset.jsonl

# Run MCP evaluation against relevance_dataset.jsonl
python -m src.evaluation.evaluate
```

## Generating the Relevance Dataset

Ideally, the relevance dataset would come from user queries and positive signals (clicks, time spent reading, bookmarks, shares, etc.). I need to figure out the best way to measure positive search signals from agents 🤔 (e.g., session length, follow-up queries, repeated searches).

Since real searches are not available so far, we auto-generate it using LLM-based evaluation.

**Step 1: Generate User Search Queries**

The system picks the most popular Hacker News stories and uses an LLM to generate realistic search queries that users might type to find those stories.

```bash
# Generate search queries for popular stories (output to /evaluation)
python -m src.evaluation.generate.generate_queries
```

The LLM prompt is defined in `src/prompts/generate_queries.md`.

Review searches manually, pick the best, and move them to the [search_queries.jsonl](src/evaluation/datasets/search_queries.jsonl).

**Step 2: Generate relevance dataset**

For each generated query:
1. Retrieves potential matches using multiple retrieval methods (text search, vector search)
2. Uses an LLM to rank the retrieved stories by relevance
3. Stores the top-ranked stories as the "relevant documents" for that query

```bash
python -m src.evaluation.generate.generate_relevance_dataset
```

The ranking prompt is defined in `src/prompts/rank_candidates.md`.

The resulting relevance dataset contains query-document pairs that represent "ideal" search results.

**Debug LLM ranked results for one query**

Check how the LLM ranks search results for a single query.

```bash
python -m src.evaluation.generate.generate_ranked_candidates --query="agent spec"
```

## Evaluating MCP Search

Once you have a relevance dataset, you can evaluate how well your MCP search performs.

```bash
# Run evaluation against current MCP implementation
python -m src.evaluation.evaluate
```

The evaluation process:
1. Takes each query from the relevance dataset
2. Runs it through the actual MCP search server
3. Compares the returned results against the relevance dataset's relevant documents
4. Calculates standard information retrieval metrics

**Improvement loop**

After each evaluation run, compare the metrics against previous runs. If metrics improve, the current MCP configuration becomes your new baseline. Then:

1. Tune your MCP search logic (adjust ranking signals, modify the search index, refine the query processing, etc.)
2. Re-run the evaluation
3. Compare metrics to see if you're getting closer to the relevance dataset
4. Repeat

