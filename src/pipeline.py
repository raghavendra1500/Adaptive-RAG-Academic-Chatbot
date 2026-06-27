"""
pipeline.py
-----------
Complete Adaptive RAG Pipeline
"""

import os
import time

from src.pdf_processor import PDFProcessor
from src.adaptive_chunker import AdaptiveChunker
from src.embedding import EmbeddingGenerator
from src.vector_database import VectorDatabase

from src.semantic_search import SemanticSearch
from src.binary_search import BinarySearchOptimizer
from src.adaptive_retriever import AdaptiveRetriever
from src.prompt_builder import PromptBuilder
from src.answer_generator import AnswerGenerator
from src.citations import CitationGenerator
from src.confidence import ConfidenceEstimator


class RAGPipeline:

    def __init__(self):

        print("=" * 60)
        print("Initializing Adaptive RAG Pipeline")
        print("=" * 60)

        # -----------------------------
        # Build Database Components
        # -----------------------------
        self.pdf_processor = PDFProcessor()
        self.chunker = AdaptiveChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_db = VectorDatabase()

        # -----------------------------
        # QA Components
        # -----------------------------
        self.search = SemanticSearch(
            embedder=self.embedder,
            vector_db=self.vector_db
        )
        self.optimizer = BinarySearchOptimizer()
        self.adaptive_retriever = AdaptiveRetriever()
        self.prompt_builder = PromptBuilder()
        self.answer_generator = AnswerGenerator()
        self.citation_generator = CitationGenerator()
        self.confidence_estimator = ConfidenceEstimator()

        print("Pipeline Ready.")
        print("=" * 60)

    def _notify(self, callback, stage, progress, message):
        if callback:
            callback(stage, progress, message)

    # =========================================================
    # Build FAISS Database
    # =========================================================

    def build_database(self, pdf_paths, progress_callback=None):
        """
        Build a FAISS database from one or more academic PDFs.
        """

        print("\nBuilding Vector Database...\n")

        if isinstance(pdf_paths, (str, os.PathLike)):
            pdf_paths = [pdf_paths]

        if not pdf_paths:
            raise ValueError("No PDF files were provided.")

        build_start = time.time()

        self._notify(
            progress_callback,
            "extracting",
            0.15,
            "Extracting PDF text"
        )

        # Step 1
        pages = []

        for pdf_path in pdf_paths:
            document_pages = self.pdf_processor.extract_text(pdf_path)

            for page in document_pages:
                page["source_file"] = os.path.basename(pdf_path)

            pages.extend(document_pages)

        print(f"Pages Extracted : {len(pages)}")

        if not pages:
            raise ValueError("No readable text was found in the PDF.")

        self._notify(
            progress_callback,
            "chunking",
            0.35,
            "Creating adaptive academic chunks"
        )

        # Step 2
        chunks = self.chunker.chunk_pages(pages)
        print(f"Chunks Created : {len(chunks)}")

        if not chunks:
            raise ValueError("No chunks were created from the PDF.")

        self._notify(
            progress_callback,
            "embedding",
            0.65,
            "Generating BAAI/bge-small-en-v1.5 embeddings"
        )

        # Step 3
        embeddings = self.embedder.generate_embeddings(chunks)
        print(f"Embeddings Generated : {len(embeddings)}")

        self._notify(
            progress_callback,
            "indexing",
            0.85,
            "Building FAISS vector index"
        )

        # Step 4
        self.vector_db.build_index(
            embeddings
        )

        self.vector_db.save_index(
            chunks
        )

        elapsed = round(time.time() - build_start, 2)

        self._notify(
            progress_callback,
            "ready",
            1.0,
            "Database ready"
        )

        print("\nVector Database Ready.\n")

        return {
            "pages": len(pages),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "vectors": self.vector_db.index.ntotal,
            "files": len(pdf_paths),
            "indexing_time": elapsed
        }

    # =========================================================
    # Ask Question
    # =========================================================

    def ask(self, question):

        start_time = time.time()

        retrieval_policy = self.adaptive_retriever.get_policy(question)

        # Semantic Retrieval
        retrieved_chunks = self.search.search(
            question,
            top_k=retrieval_policy["top_k"]
        )

        # BSCO with adaptive retrieval policy
        selected_chunks = self.optimizer.optimize(
            retrieved_chunks,
            threshold=retrieval_policy["threshold"],
            max_chunks=retrieval_policy["max_chunks"]
        )

        # Prompt
        prompt = self.prompt_builder.build_prompt(
            question,
            selected_chunks
        )

        # LLM
        answer = self.answer_generator.generate_answer(prompt)

        # Citations
        citations = self.citation_generator.generate(selected_chunks)

        # Confidence
        confidence = self.confidence_estimator.calculate(selected_chunks)

        end_time = time.time()

        return {

            "question": question,

            "answer": answer["answer"],

            "model": answer["model"],

            "citations": citations,

            "confidence": confidence,

            "retrieved_chunks": len(retrieved_chunks),

            "selected_chunks": len(selected_chunks),

            "selected_context": selected_chunks,

            "retrieval_policy": retrieval_policy,

            "response_time": round(end_time - start_time, 2)

        }

    def get_database_stats(self):
        return self.vector_db.get_stats()

    def clear_database(self):
        self.vector_db.clear()
