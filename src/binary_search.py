"""
Binary Search-Based Context Optimization
"""

import re

from config import MAX_CONTEXT_TOKENS, MIN_CONTEXT_COVERAGE, SIMILARITY_THRESHOLD


class BinarySearchOptimizer:

    def __init__(
        self,
        threshold=SIMILARITY_THRESHOLD,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        min_context_coverage=MIN_CONTEXT_COVERAGE
    ):
        self.threshold = threshold
        self.max_context_tokens = max_context_tokens
        self.min_context_coverage = min_context_coverage

    def optimize(
        self,
        retrieved_chunks,
        question="",
        threshold=None,
        max_chunks=None,
        min_chunks=1,
        query_type="open",
        return_stats=False
    ):
        """
        Find the smallest deduplicated context prefix that is likely sufficient.
        """

        if not retrieved_chunks:
            stats = self._stats([], [], 0, 0, 0)
            return ([], stats) if return_stats else []

        active_threshold = self.threshold if threshold is None else threshold
        candidates = self._deduplicate(retrieved_chunks)
        budget = min(max_chunks or len(candidates), len(candidates))
        budget = max(min_chunks, budget)
        budget = min(budget, len(candidates))

        initial_tokens = self._count_tokens(candidates)
        query_terms = self._terms(question)

        left = min(min_chunks, budget)
        right = budget
        best_size = budget

        while left <= right:
            mid = (left + right) // 2
            subset = candidates[:mid]

            if self._is_sufficient(
                subset,
                query_terms,
                active_threshold,
                min_chunks,
                query_type
            ):
                best_size = mid
                right = mid - 1
            else:
                left = mid + 1

        selected = candidates[:best_size]
        selected = self._trim_to_token_budget(selected)
        final_tokens = self._count_tokens(selected)
        stats = self._stats(
            retrieved_chunks,
            selected,
            initial_tokens,
            final_tokens,
            len(candidates)
        )

        return (selected, stats) if return_stats else selected

    def _is_sufficient(self, chunks, query_terms, threshold, min_chunks, query_type):
        if len(chunks) < min_chunks:
            return False

        average_score = sum(chunk.get("score", 0.0) for chunk in chunks) / max(len(chunks), 1)

        if average_score < max(threshold - 0.12, 0.0):
            return False

        if not query_terms:
            return True

        context_terms = set()

        for chunk in chunks:
            context_terms.update(self._terms(chunk.get("text", "")))

        coverage = len(query_terms & context_terms) / max(len(query_terms), 1)

        if query_type in {"summary", "comparison", "explanation"}:
            return coverage >= max(self.min_context_coverage - 0.10, 0.50)

        return coverage >= self.min_context_coverage

    def _deduplicate(self, chunks):
        seen = set()
        unique_chunks = []

        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.get("text", "").lower()).strip()
            fingerprint = normalized[:220]

            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            unique_chunks.append(chunk)

        return unique_chunks

    def _trim_to_token_budget(self, chunks):
        selected = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._count_tokens([chunk])

            if selected and used_tokens + chunk_tokens > self.max_context_tokens:
                break

            selected.append(chunk)
            used_tokens += chunk_tokens

        return selected or chunks[:1]

    def _count_tokens(self, chunks):
        return sum(max(1, len(chunk.get("text", "").split())) for chunk in chunks)

    def _terms(self, text):
        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
            "on", "by", "is", "are", "was", "were", "what", "which", "how",
            "why", "from", "this", "that", "into", "does", "do", "be",
        }
        return {
            word
            for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(word) > 2 and word not in stopwords
        }

    def _stats(self, retrieved_chunks, selected_chunks, initial_tokens, final_tokens, candidate_count):
        if not initial_tokens:
            initial_tokens = self._count_tokens(retrieved_chunks)

        if not final_tokens:
            final_tokens = self._count_tokens(selected_chunks)

        token_reduction = max(initial_tokens - final_tokens, 0)
        context_reduction_percent = (
            (token_reduction / initial_tokens) * 100
            if initial_tokens
            else 0.0
        )

        return {
            "initial_retrieved_chunks": len(retrieved_chunks),
            "deduplicated_candidates": candidate_count,
            "final_selected_chunks": len(selected_chunks),
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
            "token_reduction": token_reduction,
            "context_reduction_percent": round(context_reduction_percent, 2),
        }
