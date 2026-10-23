"""
Configuration module for Gas Turbine Maintenance Assistant (RAG Pipeline).
Loads settings from environment variables and provides sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# PDF Configuration
DEFAULT_PDF_NAME = "GER-3620-P.pdf"
PDF_PATH = DATA_DIR / DEFAULT_PDF_NAME

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "gas_turbine_maintenance_docs"

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# Text Splitter Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "4"))
# Chroma using cosine distance: lower is more similar.
# Similarity score threshold (0.0 to 1.0)
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "0.75"))

# LLM Configuration
# Supports "gemini" (default), "openai", or "offline"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Strict RAG constants
NOT_FOUND_RESPONSE = "Not found"
