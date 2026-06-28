"""
Post-generation grounding verification.
"""

from __future__ import annotations

import re
from typing import List


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "by", "is", "are", "was", "were", "be", "as", "that", "this", "it",
    "from", "at", "which", "their", "its", "into", "can", "may", "should",
}


def content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


class SelfVerifier:
    """Checks whether answer statements are supported by retrieved context."""

    INSUFFICIENT_MESSAGE = (
        "The uploaded academic document does not contain enough information "
        "to answer this question."
    )

    def verify(self, answer: str, chunks: List[dict]) -> dict:
        context = " ".join(chunk.get("text", "") for chunk in chunks)
        context_terms = content_terms(context)

        if not answer.strip() or not context_terms:
            return {
                "answer": self.INSUFFICIENT_MESSAGE,
                "score": 0.0,
                "unsupported_statements": [],
                "supported_statements": 0,
            }

        statements = [
            statement.strip()
            for statement in re.split(r"(?<=[.!?])\s+", answer)
            if statement.strip()
        ]

        kept = []
        unsupported = []

        for statement in statements:
            terms = content_terms(statement)

            if not terms or statement.lower().startswith(("answer:", "supporting pages:", "confidence:")):
                kept.append(statement)
                continue

            coverage = len(terms & context_terms) / max(len(terms), 1)

            if coverage >= 0.34:
                kept.append(statement)
            else:
                unsupported.append(statement)

        factual_count = max(len(statements) - 1, 1)
        score = 1.0 - (len(unsupported) / factual_count)
        score = max(0.0, min(score, 1.0))

        grounded_answer = " ".join(kept).strip()

        if not grounded_answer:
            grounded_answer = self.INSUFFICIENT_MESSAGE

        return {
            "answer": grounded_answer,
            "score": round(score, 4),
            "unsupported_statements": unsupported,
            "supported_statements": len(kept),
        }
