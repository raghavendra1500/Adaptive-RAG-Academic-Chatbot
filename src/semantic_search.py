"""
Semantic Search Module
----------------------
Encodes user queries and retrieves the most relevant chunks.
"""

import math
import re
from collections import Counter

from src.embedding import EmbeddingGenerator
from src.vector_database import VectorDatabase

from config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    HYBRID_KEYWORD_WEIGHT,
    TOP_K,
)


class SemanticSearch:

    def __init__(self, embedder=None, vector_db=None, load_existing=True):

        self.embedder = embedder or EmbeddingGenerator()

        self.vector_db = vector_db or VectorDatabase()

        if load_existing:
            try:
                self.vector_db.load_index()
            except FileNotFoundError:
                print("No vector database found yet. Process a PDF before searching.")

    def reload(self):
        self.vector_db.load_index()

    def search(self, query, top_k=TOP_K):

        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)

        # Search FAISS
        results = self.vector_db.search(
            query_embedding,
            top_k
        )

        return results

    def hybrid_search(self, query, top_k=TOP_K, candidate_multiplier=4):
        """
        Combine dense FAISS retrieval with BM25-style and keyword retrieval.
        """

        if self.vector_db.index is None or self.vector_db.metadata is None:
            self.vector_db.load_index()

        candidate_count = min(
            max(top_k * candidate_multiplier, top_k),
            self.vector_db.index.ntotal
        )
        dense_results = self.search(query, top_k=candidate_count)
        lexical_scores = self._lexical_scores(query)

        dense_by_id = {
            self._chunk_key(chunk): chunk
            for chunk in dense_results
        }

        all_candidates = {}

        for chunk in dense_results:
            all_candidates[self._chunk_key(chunk)] = chunk.copy()

        for chunk, scores in lexical_scores.items():
            if scores["bm25"] <= 0 and scores["keyword"] <= 0:
                continue

            if chunk not in all_candidates:
                all_candidates[chunk] = self.vector_db.metadata[chunk].copy()
                all_candidates[chunk]["score"] = 0.0

        if not all_candidates:
            return dense_results[:top_k]

        max_dense = max((item.get("score", 0.0) for item in all_candidates.values()), default=1.0)
        max_bm25 = max((item["bm25"] for item in lexical_scores.values()), default=1.0)

        fused = []

        for key, chunk in all_candidates.items():
            lexical = lexical_scores.get(key, {"bm25": 0.0, "keyword": 0.0})
            dense_score = chunk.get("score", 0.0) / max(max_dense, 1e-9)
            bm25_score = lexical["bm25"] / max(max_bm25, 1e-9)
            keyword_score = lexical["keyword"]

            hybrid_score = (
                HYBRID_DENSE_WEIGHT * dense_score
                + HYBRID_BM25_WEIGHT * bm25_score
                + HYBRID_KEYWORD_WEIGHT * keyword_score
            )

            chunk["dense_score"] = round(chunk.get("score", 0.0), 4)
            chunk["bm25_score"] = round(bm25_score, 4)
            chunk["keyword_score"] = round(keyword_score, 4)
            chunk["hybrid_score"] = round(hybrid_score, 4)
            chunk["score"] = round(hybrid_score, 4)
            fused.append(chunk)

        fused.sort(key=lambda item: item.get("hybrid_score", item.get("score", 0.0)), reverse=True)
        return fused[:candidate_count]

    def _lexical_scores(self, query):
        query_terms = self._tokenize(query)
        query_counts = Counter(query_terms)

        if not query_terms or not self.vector_db.metadata:
            return {}

        documents = [
            self._tokenize(chunk.get("text", ""))
            for chunk in self.vector_db.metadata
        ]
        doc_count = len(documents)
        avg_doc_len = sum(len(doc) for doc in documents) / max(doc_count, 1)

        document_frequency = Counter()

        for doc in documents:
            document_frequency.update(set(doc))

        scores = {}
        k1 = 1.5
        b = 0.75

        for index, doc in enumerate(documents):
            term_counts = Counter(doc)
            doc_len = len(doc)
            bm25 = 0.0

            for term, query_weight in query_counts.items():
                if term not in term_counts:
                    continue

                idf = math.log(1 + (doc_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                term_frequency = term_counts[term]
                denominator = term_frequency + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
                bm25 += idf * ((term_frequency * (k1 + 1)) / denominator) * query_weight

            keyword = len(set(query_terms) & set(doc)) / max(len(set(query_terms)), 1)
            scores[index] = {
                "bm25": bm25,
                "keyword": keyword,
            }

        return scores

    def _chunk_key(self, chunk):
        if self.vector_db.metadata:
            for index, metadata in enumerate(self.vector_db.metadata):
                if metadata.get("chunk_id") == chunk.get("chunk_id") and metadata.get("page_number") == chunk.get("page_number"):
                    return index

        return chunk.get("chunk_id", id(chunk))

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-zA-Z0-9]+", text.lower())
