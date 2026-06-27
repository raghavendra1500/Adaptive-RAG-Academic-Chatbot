"""
Embedding Module
----------------
Generates embeddings for text chunks and user queries.
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, DEVICE


class EmbeddingGenerator:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        print("Embedding model loaded.")

    # ===================================================
    # Document Embeddings
    # ===================================================

    def generate_embeddings(self, chunks):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    # ===================================================
    # Query Embedding
    # ===================================================

    def encode_query(self, query):

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding