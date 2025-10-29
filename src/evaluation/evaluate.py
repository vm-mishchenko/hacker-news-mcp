"""
Evaluation script for MPCRetriever.

Runs retrieval against relevance truth dataset, saves results, and compares with baseline.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.evaluation.retrievers.mpc_retriever import MPCRetriever
from src.packages.evaluation_framework import (
    QueryResult,
    Dataset,
    Evaluator,
    ExperimentConfig,
    Experiment,
    Metrics,
    Retriever,
    save_experiment,
    load_experiment,
    compare_experiments,
)

logger = logging.getLogger(__name__)

# Hardcoded mapping of retriever names to classes
RETRIEVER_MAP = {
    "mpc": MPCRetriever,
}


def parse_args():
    """Parse CLI arguments."""
    logger.info("Parsing CLI arguments")
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument("--name", help="Experiment name (default: timestamp)")
    parser.add_argument(
        "--retriever",
        choices=list(RETRIEVER_MAP.keys()),
        default="mpc",
        help=f"Retriever to use (default: mpc). Options: {', '.join(RETRIEVER_MAP.keys())}"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the experiment results (default: False)"
    )
    args = parser.parse_args()
    logger.info(f"Parsed arguments: name={args.name}, retriever={args.retriever}, save={args.save}")
    return args


def generate_experiment_name() -> str:
    """Generate timestamp-based experiment name."""
    logger.info("Generating experiment name")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"run_{timestamp}"
    logger.info(f"Generated experiment name: {name}")
    return name


def run_retrieval(dataset: Dataset, retriever: Retriever, k: int = 20) -> Dataset:
    """Run retrieval for all queries in dataset."""
    logger.info(f"Starting retrieval for {len(dataset.get_queries())} queries with k={k}")

    queries: List[QueryResult] = []
    failed_queries: List[str] = []
    total_queries = len(dataset.get_queries())

    for idx, query in enumerate(dataset.get_queries(), start=1):
        try:
            logger.info(f"Processing query {idx}/{total_queries}")
            doc_ids = retriever.retrieve(query.query, k=k)

            # Check if retriever returned None or empty list
            if doc_ids is None:
                logger.error(f"Retriever returned None for query '{query.query}'")
                failed_queries.append(query.query)
                continue

            queries.append(QueryResult(query=query.query, relevant_docs=doc_ids))
            logger.info(f"Retrieved {len(doc_ids)} documents for query: {query.query}")
        except Exception as e:
            logger.error(f"Failed to retrieve for query '{query.query}': {e}")
            failed_queries.append(query.query)

    if failed_queries:
        logger.warning(f"Failed to retrieve for {len(failed_queries)} queries: {failed_queries}")

    if len(queries) == 0:
        raise ValueError("No successful retrievals. Cannot create empty dataset.")

    results = Dataset(queries)
    logger.info(f"Retrieval complete. Processed {len(queries)} queries")
    return results


def evaluate_results(
    relevance_dataset: Dataset,
    results_dataset: Dataset,
    k_values: List[int]
) -> Metrics:
    """Evaluate retrieval results against relevance dataset."""
    logger.info("Starting evaluation")

    evaluator = Evaluator(relevance_dataset, k_values=k_values)
    metrics = evaluator.evaluate(results_dataset)

    logger.info("Evaluation complete")
    return metrics


def has_baseline() -> bool:
    """Check if baseline experiment exists."""
    logger.info("Checking for baseline")
    exists = Path("evaluation/runs/best_run/metrics.json").exists()
    logger.info(f"Baseline exists: {exists}")
    return exists


def display_final_summary(exp_name: str, metrics: Metrics, k_values: List[int]):
    """Display final experiment summary with metrics."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {exp_name}")
    logger.info("=" * 80)
    logger.info(f"MRR:     {metrics.mrr:.3f}")
    for k in k_values:
        logger.info(f"NDCG@{k}:  {metrics.ndcg_at_k[k]:.3f}")
        logger.info(f"P@{k}:     {metrics.precision_at_k[k]:.3f}")
        logger.info(f"R@{k}:     {metrics.recall_at_k[k]:.3f}")
    logger.info("=" * 80)


def print_verdict(baseline_metrics: Metrics, current_metrics: Metrics, exp_name: str):
    """Print verdict comparing baseline and current metrics."""
    logger.info("Calculating verdict")

    # Use the highest k value available
    max_k = max(current_metrics.ndcg_at_k.keys())

    # Primary metrics
    ndcg_improved = current_metrics.ndcg_at_k[max_k] > baseline_metrics.ndcg_at_k[max_k]
    ndcg_regressed = current_metrics.ndcg_at_k[max_k] < baseline_metrics.ndcg_at_k[max_k]
    mrr_improved = current_metrics.mrr > baseline_metrics.mrr
    mrr_regressed = current_metrics.mrr < baseline_metrics.mrr

    logger.info("=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if ndcg_improved and mrr_improved:
        logger.info("✓ BETTER - Primary metrics improved!")
        logger.info(
            f"  NDCG@{max_k}: {baseline_metrics.ndcg_at_k[max_k]:.3f} → {current_metrics.ndcg_at_k[max_k]:.3f}")
        logger.info(f"  MRR:     {baseline_metrics.mrr:.3f} → {current_metrics.mrr:.3f}")
        logger.info(f"  Promote: cp -r evaluation/runs/{exp_name} evaluation/runs/best_run")
    elif not ndcg_improved and not ndcg_regressed and not mrr_improved and not mrr_regressed:
        logger.info("= EQUAL - Primary metrics unchanged")
        logger.info(f"  NDCG@{max_k}: {baseline_metrics.ndcg_at_k[max_k]:.3f}")
        logger.info(f"  MRR:     {baseline_metrics.mrr:.3f}")
    elif ndcg_improved or mrr_improved:
        logger.info("≈ MIXED - Some improvements, some regressions")
    else:
        logger.info("✗ WORSE - Primary metrics regressed")

    logger.info("=" * 80)


def compare_and_decide(current_exp: Experiment):
    """Compare current experiment with baseline and print verdict."""
    logger.info(f"Comparing experiment {current_exp.config.name} with baseline")

    if not has_baseline():
        logger.warning("No baseline found. Set this as baseline:")
        logger.warning(
            f"   cp -r evaluation/runs/{current_exp.config.name} evaluation/runs/best_run")
        return

    # Load baseline experiment
    baseline_exp = load_experiment("best_run")

    # Reuse existing comparison function
    logger.info("=" * 80)
    compare_experiments(baseline_exp, current_exp)

    # Print verdict
    print_verdict(baseline_exp.metrics, current_exp.metrics, current_exp.config.name)


def run():
    """Main coordinator function."""
    logger.info("Starting evaluation script")

    # Parse arguments
    args = parse_args()
    exp_name = args.name if args.name else generate_experiment_name()
    retriever_name = args.retriever

    # Hardcoded k_values
    # k_values = [1, 5, 10]
    k_values = [5]
    logger.info(f"Using k_values: {k_values}")

    # Load relevance truth dataset from JSONL file
    logger.info("Loading relevance truth dataset")
    relevance_dataset_path = Path(__file__).parent / "datasets" / "relevance_dataset.jsonl"
    relevance_dataset = Dataset.from_jsonl(str(relevance_dataset_path))

    # Initialize retriever from mapping
    retriever_class = RETRIEVER_MAP[retriever_name]
    logger.info(f"Initializing {retriever_class.__name__}")
    retriever = retriever_class()

    # Run retrieval
    results_dataset = run_retrieval(relevance_dataset, retriever, k=20)

    # Evaluate
    current_metrics = evaluate_results(relevance_dataset, results_dataset, k_values)

    # Create experiment config
    current_config = ExperimentConfig(
        timestamp=datetime.now(timezone.utc),
        retriever=retriever_name,
        name=exp_name,
        retriever_config={},
        metadata={
            "k_values": k_values,
            "dataset_size": len(relevance_dataset.get_queries()),
        },
    )

    # Create current experiment object
    current_exp = Experiment(
        dataset=results_dataset,
        metrics=current_metrics,
        config=current_config
    )

    # Save (only if --save flag is passed)
    if args.save:
        save_experiment(exp_name, results_dataset, current_metrics, current_config)
        logger.info(f"Saved to: evaluation/runs/{exp_name}/")
    else:
        logger.info("Skipping save (use --save to save results)")

    # Display final summary at the end
    display_final_summary(exp_name, current_metrics, k_values)

    # Compare with baseline
    compare_and_decide(current_exp)

    logger.info("Evaluation script complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run()
