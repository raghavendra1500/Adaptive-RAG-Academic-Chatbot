"""
Adaptive retrieval policy for academic RAG.

The policy classifies the question and adapts how many chunks to retrieve,
how strict the similarity threshold should be, and how much context should be
sent to the answer generator.
"""

from src.query_classifier import QueryClassifier


class AdaptiveRetriever:

    def __init__(self):
        self.classifier = QueryClassifier()
        self.policies = {
            "definition": {
                "top_k": 6,
                "threshold": 0.68,
                "max_chunks": 3,
                "min_chunks": 1
            },
            "factual": {
                "top_k": 6,
                "threshold": 0.70,
                "max_chunks": 3,
                "min_chunks": 1
            },
            "summary": {
                "top_k": 12,
                "threshold": 0.60,
                "max_chunks": 6,
                "min_chunks": 3
            },
            "comparison": {
                "top_k": 12,
                "threshold": 0.58,
                "max_chunks": 7,
                "min_chunks": 3
            },
            "methodology": {
                "top_k": 10,
                "threshold": 0.62,
                "max_chunks": 5,
                "min_chunks": 2
            },
            "explanation": {
                "top_k": 10,
                "threshold": 0.62,
                "max_chunks": 5,
                "min_chunks": 2
            },
            "open": {
                "top_k": 10,
                "threshold": 0.62,
                "max_chunks": 5,
                "min_chunks": 2
            }
        }

    def classify_question(self, question):
        """
        Classify common academic question types with transparent rules.
        """

        return self.classifier.classify(question).query_type

    def get_policy(self, question, retrieval_signals=None):
        profile = self.classifier.classify(question)
        query_type = profile.query_type
        policy = self.policies[query_type].copy()

        complexity_boost = round(profile.complexity * 4)
        policy["top_k"] += complexity_boost
        policy["max_chunks"] += 1 if profile.complexity >= 0.7 else 0

        if profile.has_follow_up:
            policy["top_k"] += 2
            policy["max_chunks"] += 1

        if retrieval_signals:
            spread = retrieval_signals.get("similarity_spread", 0.0)
            density = retrieval_signals.get("document_density", 0.0)

            if spread < 0.08:
                policy["top_k"] += 2

            if density > 0.70:
                policy["top_k"] += 1

        policy["top_k"] = min(max(policy["top_k"], 4), 20)
        policy["max_chunks"] = min(max(policy["max_chunks"], policy["min_chunks"]), 9)
        policy["query_type"] = query_type
        policy["complexity"] = profile.complexity
        policy["query_tokens"] = profile.token_count
        policy["has_follow_up"] = profile.has_follow_up
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
