"""
Adaptive retrieval policy for academic RAG.

The policy classifies the question and adapts how many chunks to retrieve,
how strict the similarity threshold should be, and how much context should be
sent to the answer generator.
"""

import re


class AdaptiveRetriever:

    def __init__(self):
        self.policies = {
            "definition": {
                "top_k": 6,
                "threshold": 0.68,
                "max_chunks": 3
            },
            "factual": {
                "top_k": 6,
                "threshold": 0.70,
                "max_chunks": 3
            },
            "summary": {
                "top_k": 12,
                "threshold": 0.60,
                "max_chunks": 6
            },
            "comparison": {
                "top_k": 12,
                "threshold": 0.58,
                "max_chunks": 7
            },
            "methodology": {
                "top_k": 10,
                "threshold": 0.62,
                "max_chunks": 5
            },
            "open": {
                "top_k": 10,
                "threshold": 0.62,
                "max_chunks": 5
            }
        }

    def classify_question(self, question):
        """
        Classify common academic question types with transparent rules.
        """

        normalized = question.lower().strip()

        if re.search(r"\b(compare|contrast|difference|similarities|versus| vs )\b", normalized):
            return "comparison"

        if re.search(r"\b(summarize|summary|overview|main idea|key points|abstract)\b", normalized):
            return "summary"

        if re.search(r"\b(method|methodology|approach|algorithm|experiment|evaluation|procedure)\b", normalized):
            return "methodology"

        if re.search(r"\b(define|definition|what is|what are|meaning of)\b", normalized):
            return "definition"

        if re.search(r"\b(when|where|who|which|how many|list|name|state)\b", normalized):
            return "factual"

        return "open"

    def get_policy(self, question):
        query_type = self.classify_question(question)
        policy = self.policies[query_type].copy()
        policy["query_type"] = query_type
        return policy

    def select_chunks(self, retrieved_chunks, policy):
        """
        Keep chunks above the adaptive threshold, with a small fallback to the
        best chunk so the model can still say when evidence is insufficient.
        """

        if not retrieved_chunks:
            return []

        selected = [
            chunk
            for chunk in retrieved_chunks
            if chunk.get("score", 0.0) >= policy["threshold"]
        ]

        if not selected:
            selected = retrieved_chunks[:1]

        return selected[:policy["max_chunks"]]
