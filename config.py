"""
Configuration file for Adaptive RAG Academic Chatbot
"""

import os

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTOR_DB_FOLDER = os.path.join(BASE_DIR, "vector_db")
PROCESSED_DATA_FOLDER = os.path.join(BASE_DIR, "processed_data")

# Create folders automatically if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

# ==========================================================
# PDF Processing
# ==========================================================

SUPPORTED_FILE_TYPES = ["pdf"]

# ==========================================================
# Adaptive Chunking
# ==========================================================

MIN_CHUNK_SIZE = 150
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEVICE='cpu'


# ==========================
# Vector Database
# ==========================

VECTOR_DB_FOLDER = "vector_db"

FAISS_INDEX_FILE = "vector_db/faiss_index.bin"

METADATA_FILE = "vector_db/chunk_metadata.pkl"

TOP_K = 5


# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 10

# ==========================================================
# Binary Search
# ==========================================================

SIMILARITY_THRESHOLD = 0.70
MAX_CONTEXT_TOKENS = 1400
MIN_CONTEXT_COVERAGE = 0.62

# ==========================================================
# Hybrid Retrieval and Reranking
# ==========================================================

HYBRID_DENSE_WEIGHT = 0.55
HYBRID_BM25_WEIGHT = 0.30
HYBRID_KEYWORD_WEIGHT = 0.15
RERANK_TOP_N = 12
CROSS_ENCODER_MODEL = None

# ==========================================================
# Ollama
# ==========================================================

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

# ==========================================================
# Streamlit
# ==========================================================

APP_TITLE = "Adaptive RAG Academic Chatbot"

APP_ICON = "📚"

# ==========================================================
# Confidence Score
# ==========================================================

MIN_CONFIDENCE = 0.50

# ==========================================================
# Research Experiments
# ==========================================================

EXPERIMENT_NAME = "Adaptive_RAG_v1"

SAVE_RETRIEVAL_LOGS = True

SAVE_RESPONSE_LOGS = True
