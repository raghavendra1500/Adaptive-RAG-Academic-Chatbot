"""
Candidate reranking after hybrid retrieval.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config import CROSS_ENCODER_MODEL


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class CrossEncoderReranker:
    """
    Rerank retrieval candidates.

    A configured cross-encoder is used when available. Otherwise, the fallback
    combines hybrid score with transparent lexical agreement so the project
    remains runnable offline.
    """

    def __init__(self, model_name: Optional[str] = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self.model = None

        if model_name:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(model_name)
            except Exception as exc:
                print(f"Cross-encoder unavailable, using heuristic reranker: {exc}")

    def rerank(self, query: str, chunks: List[dict], top_n: int) -> List[dict]:
        if not chunks:
            return []

        candidates = [chunk.copy() for chunk in chunks]

        if self.model is not None:
            pairs = [(query, chunk.get("text", "")) for chunk in candidates]
            scores = self.model.predict(pairs)

            for chunk, score in zip(candidates, scores):
                chunk["cross_encoder_score"] = float(score)
                chunk["rerank_score"] = float(score)
        else:
            query_terms = set(tokenize(query))

            for chunk in candidates:
                text_terms = set(tokenize(chunk.get("text", "")))
                overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
                hybrid_score = chunk.get("hybrid_score", chunk.get("score", 0.0))
                chunk["cross_encoder_score"] = round(overlap, 4)
                chunk["rerank_score"] = round((0.7 * hybrid_score) + (0.3 * overlap), 4)

        candidates.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_n]
