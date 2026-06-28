"""
Runtime evaluation metrics for retrieval and context optimization.
"""

from __future__ import annotations

from typing import List


class EvaluationTracker:
    """Computes observable metrics without requiring labeled datasets."""

    def calculate(
        self,
        retrieved_chunks: List[dict],
        selected_chunks: List[dict],
        bsco_stats: dict,
        response_time: float,
        answer: str,
    ) -> dict:
        retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks]
        selected_ids = {chunk.get("chunk_id") for chunk in selected_chunks}

        first_selected_rank = None
        for index, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in selected_ids:
                first_selected_rank = index
                break

        mrr = 0.0 if first_selected_rank is None else 1.0 / first_selected_rank
        precision_at_selected = len(selected_ids) / max(len(retrieved_chunks), 1)
        recall_proxy = len(selected_ids) / max(bsco_stats.get("initial_retrieved_chunks", 1), 1)

        return {
            "retrieval_precision_proxy": round(precision_at_selected, 4),
            "retrieval_recall_proxy": round(recall_proxy, 4),
            "mrr_proxy": round(mrr, 4),
            "latency_seconds": response_time,
            "prompt_tokens": bsco_stats.get("final_tokens", 0),
            "response_tokens": len(answer.split()),
            "context_compression_ratio": bsco_stats.get("context_reduction_percent", 0.0),
        }
