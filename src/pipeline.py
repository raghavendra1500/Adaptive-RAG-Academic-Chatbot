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
from src.reranker import CrossEncoderReranker
from src.self_verifier import SelfVerifier
from src.evaluation import EvaluationTracker
from config import RERANK_TOP_N


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
        self.reranker = CrossEncoderReranker()
        self.verifier = SelfVerifier()
        self.evaluator = EvaluationTracker()
        self.conversation_history = []

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

        # Hybrid Retrieval: dense FAISS + BM25-style lexical + keyword scores
        retrieved_chunks = self.search.hybrid_search(
            question,
            top_k=retrieval_policy["top_k"]
        )

        similarity_scores = [chunk.get("score", 0.0) for chunk in retrieved_chunks]
        retrieval_signals = {
            "similarity_spread": (
                max(similarity_scores) - min(similarity_scores)
                if similarity_scores
                else 0.0
            ),
            "document_density": len(retrieved_chunks) / max(self.vector_db.get_stats()["metadata_count"], 1),
        }
        retrieval_policy = self.adaptive_retriever.get_policy(question, retrieval_signals)

        reranked_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_n=min(RERANK_TOP_N, len(retrieved_chunks))
        )

        # BSCO with adaptive retrieval policy
        selected_chunks, bsco_stats = self.optimizer.optimize(
            reranked_chunks,
            question=question,
            threshold=retrieval_policy["threshold"],
            max_chunks=retrieval_policy["max_chunks"],
            min_chunks=retrieval_policy["min_chunks"],
            query_type=retrieval_policy["query_type"],
            return_stats=True
        )

        # Prompt
        prompt = self.prompt_builder.build_prompt(
            question,
            selected_chunks,
            query_type=retrieval_policy["query_type"],
            conversation_history=self.conversation_history
        )

        # LLM
        answer = self.answer_generator.generate_answer(prompt)

        # Citations
        citations = self.citation_generator.generate(selected_chunks)

        # Self verification
        verification = self.verifier.verify(answer["answer"], selected_chunks)
        verified_answer = verification["answer"]

        # Confidence
        confidence = self.confidence_estimator.calculate(
            selected_chunks,
            citations=citations,
            verification=verification
        )

        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        evaluation = self.evaluator.calculate(
            reranked_chunks,
            selected_chunks,
            bsco_stats,
            response_time,
            verified_answer
        )

        result = {

            "question": question,

            "answer": verified_answer,

            "model": answer["model"],

            "citations": citations,

            "confidence": confidence,

            "retrieved_chunks": len(retrieved_chunks),

            "selected_chunks": len(selected_chunks),

            "selected_context": selected_chunks,

            "retrieval_policy": retrieval_policy,

            "bsco": bsco_stats,

            "verification": verification,

            "evaluation": evaluation,

            "response_time": response_time

        }

        self.conversation_history.append(
            {
                "question": question,
                "answer": verified_answer,
                "confidence": confidence,
            }
        )
        self.conversation_history = self.conversation_history[-5:]

        return result

    def get_database_stats(self):
        return self.vector_db.get_stats()

    def clear_database(self):
        self.vector_db.clear()
        self.conversation_history = []
