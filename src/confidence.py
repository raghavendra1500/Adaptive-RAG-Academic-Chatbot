"""
confidence.py
-------------
Calculates confidence score for retrieved document chunks.
"""


class ConfidenceEstimator:

    def __init__(self):
        pass

    def calculate(self, retrieved_chunks):
        """
        Calculate confidence based on similarity scores.

        Returns:
            {
                "score": float,
                "level": str
            }
        """

        if not retrieved_chunks:
            return {
                "score": 0.0,
                "level": "Low"
            }

        scores = [
            chunk["score"]
            for chunk in retrieved_chunks
            if "score" in chunk
        ]

        if not scores:
            return {
                "score": 0.0,
                "level": "Low"
            }

        average_score = sum(scores) / len(scores)

        if average_score >= 0.80:
            level = "High"
        elif average_score >= 0.65:
            level = "Medium"
        else:
            level = "Low"

        return {
            "score": round(average_score, 4),
            "level": level
        }