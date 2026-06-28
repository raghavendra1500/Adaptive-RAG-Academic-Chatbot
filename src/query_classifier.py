"""
Transparent query classification for adaptive academic retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryProfile:
    """Structured description of a user question."""

    query_type: str
    complexity: float
    token_count: int
    has_follow_up: bool


class QueryClassifier:
    """Rule-based classifier used to keep routing explainable."""

    FOLLOW_UP_TERMS = {
        "it",
        "this",
        "that",
        "they",
        "them",
        "those",
        "above",
        "previous",
        "same",
    }

    def classify(self, question: str) -> QueryProfile:
        normalized = question.lower().strip()
        words = re.findall(r"[a-zA-Z0-9]+", normalized)
        token_count = len(words)

        if re.search(r"\b(compare|contrast|difference|similarities|versus| vs )\b", normalized):
            query_type = "comparison"
        elif re.search(r"\b(summarize|summary|overview|main idea|key points|abstract)\b", normalized):
            query_type = "summary"
        elif re.search(r"\b(method|methodology|approach|algorithm|experiment|evaluation|procedure|steps)\b", normalized):
            query_type = "methodology"
        elif re.search(r"\b(define|definition|what is|what are|meaning of)\b", normalized):
            query_type = "definition"
        elif re.search(r"\b(when|where|who|which|how many|list|name|state)\b", normalized):
            query_type = "factual"
        elif re.search(r"\b(why|how|explain|discuss|analyze|relationship|impact)\b", normalized):
            query_type = "explanation"
        else:
            query_type = "open"

        complexity = 0.25
        complexity += min(token_count / 42, 0.35)
        complexity += 0.18 if query_type in {"summary", "comparison", "explanation"} else 0.0
        complexity += 0.12 if len(re.findall(r"\b(and|or|across|between|multiple)\b", normalized)) else 0.0
        complexity = round(min(complexity, 1.0), 3)

        has_follow_up = bool(set(words) & self.FOLLOW_UP_TERMS)

        return QueryProfile(
            query_type=query_type,
            complexity=complexity,
            token_count=token_count,
            has_follow_up=has_follow_up,
        )
