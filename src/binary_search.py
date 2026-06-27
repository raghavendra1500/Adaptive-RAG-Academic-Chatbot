"""
Binary Search-Based Context Optimization
"""

from config import SIMILARITY_THRESHOLD


class BinarySearchOptimizer:

    def __init__(self, threshold=SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def optimize(self, retrieved_chunks, threshold=None, max_chunks=None):
        """
        Select the smallest high-relevance prefix from sorted retrieval results.

        FAISS returns chunks in descending score order. The binary search finds
        the last chunk above the adaptive threshold, then caps the result to the
        requested context budget.
        """

        if not retrieved_chunks:
            return []

        active_threshold = self.threshold if threshold is None else threshold

        left = 0
        right = len(retrieved_chunks) - 1

        boundary = -1

        while left <= right:

            mid = (left + right) // 2

            score = retrieved_chunks[mid]["score"]

            if score >= active_threshold:
                boundary = mid
                left = mid + 1
            else:
                right = mid - 1

        if boundary == -1:
            selected = retrieved_chunks[:1]
        else:
            selected = retrieved_chunks[:boundary + 1]

        if max_chunks is not None:
            selected = selected[:max_chunks]

        return selected
