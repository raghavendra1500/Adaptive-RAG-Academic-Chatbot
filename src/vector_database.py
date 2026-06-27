"""
vector_database.py
------------------
FAISS Vector Database
"""

import os
import pickle
import faiss
import numpy as np

from config import (
    FAISS_INDEX_FILE,
    METADATA_FILE
)


class VectorDatabase:

    def __init__(self):

        self.index = None
        self.metadata = None

    # =====================================================
    # Build FAISS Index
    # =====================================================

    def build_index(self, embeddings):

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Indexed {self.index.ntotal} vectors.")

    # =====================================================
    # Save Database
    # =====================================================

    def save(self, chunks):

        os.makedirs(
            os.path.dirname(FAISS_INDEX_FILE),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            FAISS_INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "wb"
        ) as f:

            pickle.dump(chunks, f)

        self.metadata = chunks

        print("Vector database saved.")

    def save_index(self, chunks):
        """
        Backward-compatible wrapper used by scripts and the pipeline.
        """

        self.save(chunks)

    # =====================================================
    # Load Database
    # =====================================================

    def load(self):

        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                "Vector database files were not found. Process a PDF before searching."
            )

        self.index = faiss.read_index(
            FAISS_INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print("Vector database loaded.")

    def load_index(self):
        """
        Backward-compatible wrapper used by search modules.
        """

        self.load()

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query_embedding,
        top_k=10
    ):

        if self.index is None or self.metadata is None:
            raise RuntimeError(
                "Vector database is not loaded. Process a PDF before asking questions."
            )

        top_k = min(top_k, self.index.ntotal)

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            chunk = self.metadata[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    def get_stats(self):
        """
        Return simple index statistics for UI and experiments.
        """

        vector_count = 0

        if self.index is not None:
            vector_count = self.index.ntotal

        metadata_count = 0

        if self.metadata is not None:
            metadata_count = len(self.metadata)

        pages = set()

        if self.metadata:
            pages = {
                item.get("page_number")
                for item in self.metadata
                if item.get("page_number") is not None
            }

        return {
            "vector_count": vector_count,
            "metadata_count": metadata_count,
            "page_count": len(pages)
        }

    def clear(self):
        """
        Remove persisted FAISS and metadata files.
        """

        for file_path in [FAISS_INDEX_FILE, METADATA_FILE]:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.index = None
        self.metadata = None
