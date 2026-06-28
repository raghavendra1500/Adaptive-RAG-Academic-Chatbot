"""
confidence.py
-------------
Calculates confidence score for retrieved document chunks.
"""


class ConfidenceEstimator:

    def __init__(self):
        pass

    def calculate(self, retrieved_chunks, citations=None, verification=None):
        """
        Calculate confidence from retrieval, citation, agreement, reranking, and verification.

        Returns:
            {
                "score": float,
                "level": str
            }
        """

        if not retrieved_chunks:
            return {
                "score": 0.0,
                "level": "Low",
                "explanation": "No retrieved evidence was available."
            }

        scores = [
            chunk["score"]
            for chunk in retrieved_chunks
            if "score" in chunk
        ]

        if not scores:
            return {
                "score": 0.0,
                "level": "Low",
                "explanation": "Retrieved chunks did not include usable scores."
            }

        retrieval_score = sum(scores) / len(scores)
        citation_coverage = self._citation_coverage(retrieved_chunks, citations)
        agreement_score = self._chunk_agreement(retrieved_chunks)
        rerank_score = self._average(
            [
                chunk.get("cross_encoder_score", chunk.get("rerank_score", 0.0))
                for chunk in retrieved_chunks
            ]
        )
        verification_score = (verification or {}).get("score", 0.0)

        score = (
            0.30 * retrieval_score
            + 0.20 * citation_coverage
            + 0.18 * agreement_score
            + 0.17 * rerank_score
            + 0.15 * verification_score
        )

        if score >= 0.78:
            level = "High"
        elif score >= 0.58:
            level = "Medium"
        else:
            level = "Low"

        return {
            "score": round(score, 4),
            "level": level,
            "components": {
                "retrieval_similarity": round(retrieval_score, 4),
                "citation_coverage": round(citation_coverage, 4),
                "chunk_agreement": round(agreement_score, 4),
                "cross_encoder": round(rerank_score, 4),
                "verification": round(verification_score, 4),
            },
            "explanation": (
                "Confidence combines retrieval similarity, citation coverage, "
                "agreement across selected chunks, reranking score, and "
                "post-generation verification."
            )
        }

    def _citation_coverage(self, chunks, citations):
        pages = set((citations or {}).get("pages", []))

        if not chunks:
            return 0.0

        cited_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("page_number") in pages
        ]

        return len(cited_chunks) / len(chunks)

    def _chunk_agreement(self, chunks):
        if len(chunks) <= 1:
            return 1.0

        term_sets = [
            {
                word.lower()
                for word in chunk.get("text", "").split()
                if len(word) > 3
            }
            for chunk in chunks
        ]

        overlaps = []

        for index, terms in enumerate(term_sets):
            others = set().union(*(term_sets[:index] + term_sets[index + 1:]))
            overlaps.append(len(terms & others) / max(len(terms), 1))

        return self._average(overlaps)

    def _average(self, values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else 0.0
