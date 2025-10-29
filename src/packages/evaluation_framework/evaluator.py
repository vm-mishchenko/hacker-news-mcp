"""
Evaluator for computing retrieval metrics.
"""

import logging
import math
from typing import List, Dict, Set

from .dataset import Dataset
from .models import DocumentId, Metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Compares retrieval results against ground truth."""

    def __init__(self, dataset: Dataset, k_values: List[int] = None):
        """Initialize evaluator with ground truth dataset."""
        logger.info("Initializing evaluator")
        self.dataset = dataset
        self.k_values = k_values or [1, 5, 10]
        logger.info(f"Evaluator initialized with k_values={self.k_values}")

    def evaluate(self, results: Dataset) -> Metrics:
        """Evaluate retrieval results against ground truth."""
        logger.info(f"Starting evaluation for {len(results.get_queries())} queries")

        precision_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        recall_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        ndcg_at_k: Dict[int, List[float]] = {k: [] for k in self.k_values}
        mrr_scores: List[float] = []
        per_query_scores: Dict[str, Dict[str, float]] = {}

        for query_result in results.get_queries():
            query_text = query_result.query
            retrieved_docs = query_result.relevant_docs
            query = self.dataset.get_query(query_text)

            # Skip queries with no relevant docs
            if len(query.relevant_docs) == 0:
                logger.warning(f"Skipping query '{query_text}' - no relevant docs")
                continue

            # Build set of relevant doc IDs
            relevant_doc_ids = set(query.relevant_docs)

            # Calculate metrics for each k
            query_scores = {}
            for k in self.k_values:
                # Precision@K
                p_at_k = self._calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                precision_at_k[k].append(p_at_k)
                query_scores[f"precision@{k}"] = p_at_k

                # Recall@K
                r_at_k = self._calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                recall_at_k[k].append(r_at_k)
                query_scores[f"recall@{k}"] = r_at_k

                # NDCG@K
                ndcg = self._calculate_ndcg_at_k(
                    retrieved_docs, relevant_doc_ids, k
                )
                ndcg_at_k[k].append(ndcg)
                query_scores[f"ndcg@{k}"] = ndcg

            # MRR
            mrr = self._calculate_mrr(retrieved_docs, relevant_doc_ids)
            mrr_scores.append(mrr)
            query_scores["mrr"] = mrr

            per_query_scores[query_text] = query_scores

        # Aggregate metrics
        metrics = Metrics(
            precision_at_k={k: self._mean(scores) for k, scores in precision_at_k.items()},
            recall_at_k={k: self._mean(scores) for k, scores in recall_at_k.items()},
            ndcg_at_k={k: self._mean(scores) for k, scores in ndcg_at_k.items()},
            mrr=self._mean(mrr_scores),
            per_query_scores=per_query_scores
        )

        logger.info("Evaluation complete")
        return metrics

    def _calculate_precision_at_k(
        self,
        retrieved: List[DocumentId],
        relevant_doc_ids: Set[DocumentId],
        k: int
    ) -> float:
        """Calculate Precision@K."""
        retrieved_at_k = retrieved[:k]
        if len(retrieved_at_k) == 0:
            return 0.0

        relevant_count = sum(
            1 for doc_id in retrieved_at_k
            if doc_id in relevant_doc_ids
        )
        return relevant_count / len(retrieved_at_k)

    def _calculate_recall_at_k(
        self,
        retrieved: List[DocumentId],
        relevant_doc_ids: Set[DocumentId],
        k: int
    ) -> float:
        """Calculate Recall@K."""
        retrieved_at_k = retrieved[:k]

        total_relevant = len(relevant_doc_ids)
        if total_relevant == 0:
            return 0.0

        relevant_retrieved = sum(
            1 for doc_id in retrieved_at_k
            if doc_id in relevant_doc_ids
        )
        return relevant_retrieved / total_relevant

    def _calculate_ndcg_at_k(
        self,
        retrieved: List[DocumentId],
        relevant_doc_ids: Set[DocumentId],
        k: int
    ) -> float:
        """Calculate NDCG@K using binary relevance."""
        retrieved_at_k = retrieved[:k]

        # Calculate DCG (binary relevance: 1 if relevant, 0 otherwise)
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_at_k, 1):
            rel = 1 if doc_id in relevant_doc_ids else 0
            dcg += rel / math.log2(i + 1)

        # Calculate IDCG (ideal DCG)
        # In binary relevance, ideal is all relevant docs at the top
        num_relevant_in_top_k = min(len(relevant_doc_ids), k)
        idcg = 0.0
        for i in range(1, num_relevant_in_top_k + 1):
            idcg += 1.0 / math.log2(i + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_mrr(
        self,
        retrieved: List[DocumentId],
        relevant_doc_ids: Set[DocumentId]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_doc_ids:
                return 1.0 / i
        return 0.0

    def _mean(self, values: List[float]) -> float:
        """Calculate mean of values."""
        if len(values) == 0:
            return 0.0
        return sum(values) / len(values)

    def show_failures(self, results: Dataset, n: int = 5) -> None:
        """Print worst N queries with details."""
        logger.info(f"Analyzing top {n} failures")

        # Calculate NDCG@10 for each query
        query_ndcg = []
        for query_result in results.get_queries():
            query_text = query_result.query
            retrieved_docs = query_result.relevant_docs
            query = self.dataset.get_query(query_text)

            if len(query.relevant_docs) == 0:
                continue

            relevant_doc_ids = set(query.relevant_docs)

            ndcg = self._calculate_ndcg_at_k(retrieved_docs, relevant_doc_ids, 10)
            query_ndcg.append((query_text, ndcg, query, retrieved_docs, relevant_doc_ids))

        # Sort by NDCG ascending (worst first)
        query_ndcg.sort(key=lambda x: x[1])

        # Log worst N
        logger.info("=" * 80)
        logger.info(f"WORST {n} QUERIES (by NDCG@10)")
        logger.info("=" * 80)

        for i, (query_text, ndcg, query, retrieved, relevant_doc_ids) in enumerate(query_ndcg[:n],
                                                                                   1):
            logger.info(f"{i}. Query: {query_text}")
            logger.info(f"   NDCG@10: {ndcg:.3f}")
            logger.info("   Expected docs:")
            for doc_id in query.relevant_docs:
                logger.info(f"     - {doc_id}")

            logger.info("   Retrieved docs (top 10):")
            for j, doc_id in enumerate(retrieved[:10], 1):
                marker = "✓" if doc_id in relevant_doc_ids else "✗"
                logger.info(f"     {j}. {doc_id} {marker}")

            logger.info("-" * 80)

        logger.info("Failure analysis complete")
