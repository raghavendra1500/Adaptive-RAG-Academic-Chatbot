"""
Semantic Search Module
----------------------
Encodes user queries and retrieves the most relevant chunks.
"""

from src.embedding import EmbeddingGenerator
from src.vector_database import VectorDatabase

from config import TOP_K


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
