"""
Experiment management for saving, loading, and comparing experiments.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .dataset import Dataset
# i need to use absolute imports - just for the consistency
from .models import Metrics, ExperimentConfig, Experiment

logger = logging.getLogger(__name__)


def save_experiment(
    exp_name: str,
    results: Dataset,
    metrics: Metrics,
    config: ExperimentConfig
) -> None:
    """Save experiment results, metrics, and config to disk."""
    logger.info(f"Saving experiment: {exp_name}")

    # Create experiment directory
    exp_dir = Path("evaluation/runs") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSONL
    results_path = exp_dir / "results.jsonl"

    # Save dataset queries with their relevant docs
    with open(results_path, 'w', encoding='utf-8') as f:
        for query in results.get_queries():
            f.write(json.dumps(asdict(query)) + '\n')

    logger.info(f"Saved results to {results_path}")

    # Save metrics as JSON
    metrics_path = exp_dir / "metrics.json"
    metrics_dict = {
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "ndcg_at_k": metrics.ndcg_at_k,
        "mrr": metrics.mrr,
        "per_query_scores": metrics.per_query_scores
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")

    # Save config as JSON
    config_path = exp_dir / "config.json"
    config_dict = {
        "timestamp": config.timestamp.isoformat(),
        "retriever": config.retriever,
        "retriever_config": config.retriever_config,
        "metadata": config.metadata
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Saved config to {config_path}")
    logger.info(f"Experiment {exp_name} saved successfully")


def load_experiment(exp_name: str) -> Experiment:
    """Load experiment results, metrics, and config from disk."""
    logger.info(f"Loading experiment: {exp_name}")

    exp_dir = Path("evaluation/runs") / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load results
    results_path = exp_dir / "results.jsonl"
    results = Dataset.from_jsonl(str(results_path))

    # Load metrics
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics_dict = json.load(f)

    # Convert string keys to int for k_values
    precision_at_k = {int(k): v for k, v in metrics_dict["precision_at_k"].items()}
    recall_at_k = {int(k): v for k, v in metrics_dict["recall_at_k"].items()}
    ndcg_at_k = {int(k): v for k, v in metrics_dict["ndcg_at_k"].items()}

    metrics = Metrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
        mrr=metrics_dict["mrr"],
        per_query_scores=metrics_dict["per_query_scores"]
    )

    logger.info(f"Loaded metrics from {metrics_path}")

    # Load config
    config_path = exp_dir / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    config = ExperimentConfig(
        timestamp=datetime.fromisoformat(config_dict["timestamp"]),
        retriever=config_dict["retriever"],
        name=exp_name,  # Use directory name as experiment name
        retriever_config=config_dict.get("retriever_config", {}),
        metadata=config_dict.get("metadata", {})
    )

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Experiment {exp_name} loaded successfully")

    return Experiment(
        dataset=results,
        metrics=metrics,
        config=config
    )


def compare_experiments(exp1: Experiment, exp2: Experiment) -> None:
    """Compare two experiments and print delta metrics."""
    logger.info(f"Comparing experiments: {exp1.config.name} vs {exp2.config.name}")

    metrics1 = exp1.metrics
    metrics2 = exp2.metrics

    # Log comparison
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Experiment 1: {exp1.config.name}")
    logger.info(f"Experiment 2: {exp2.config.name}")
    logger.info("=" * 80)

    # Compare MRR
    mrr1 = metrics1.mrr
    mrr2 = metrics2.mrr
    mrr_delta = mrr2 - mrr1
    mrr_pct = (mrr_delta / mrr1 * 100) if mrr1 != 0 else 0

    logger.info(f"{'Metric':<20} {'Exp1':>10} {'Exp2':>10} {'Delta':>12} {'% Change':>12}")
    logger.info("-" * 80)
    logger.info(f"{'MRR':<20} {mrr1:>10.3f} {mrr2:>10.3f} {mrr_delta:>+12.3f} {mrr_pct:>+11.1f}%")

    # Compare metrics at each k
    k_values = sorted(metrics1.ndcg_at_k.keys())

    for k in k_values:
        # NDCG@K
        ndcg1 = metrics1.ndcg_at_k[k]
        ndcg2 = metrics2.ndcg_at_k[k]
        ndcg_delta = ndcg2 - ndcg1
        ndcg_pct = (ndcg_delta / ndcg1 * 100) if ndcg1 != 0 else 0

        logger.info(
            f"{'NDCG@' + str(k):<20} {ndcg1:>10.3f} {ndcg2:>10.3f} {ndcg_delta:>+12.3f} {ndcg_pct:>+11.1f}%")

        # Precision@K
        p1 = metrics1.precision_at_k[k]
        p2 = metrics2.precision_at_k[k]
        p_delta = p2 - p1
        p_pct = (p_delta / p1 * 100) if p1 != 0 else 0

        logger.info(
            f"{'Precision@' + str(k):<20} {p1:>10.3f} {p2:>10.3f} {p_delta:>+12.3f} {p_pct:>+11.1f}%")

        # Recall@K
        r1 = metrics1.recall_at_k[k]
        r2 = metrics2.recall_at_k[k]
        r_delta = r2 - r1
        r_pct = (r_delta / r1 * 100) if r1 != 0 else 0

        logger.info(
            f"{'Recall@' + str(k):<20} {r1:>10.3f} {r2:>10.3f} {r_delta:>+12.3f} {r_pct:>+11.1f}%")

    logger.info("=" * 80)

    logger.info("Experiment comparison complete")
