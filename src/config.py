"""
Configuration settings for BhuvanGeo-RAG.
Handles paths, model choices, and pipeline parameters.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
DOCS_DIR = BASE_DIR / "docs"

# Ensure all directories exist
for d in [DATA_DIR, DOCUMENTS_DIR, PROCESSED_DIR, CHROMA_DB_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Chunking & Embedding Parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "isro_bhuvan_knowledge_base"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 4

# Quality / Confidence Thresholds
SIMILARITY_DISTANCE_THRESHOLD = 1.25  # Lower L2 / cosine distance is closer in Chroma

# Satellite Mission metadata constants
SUPPORTED_MISSIONS = [
    "Resourcesat-1",
    "Resourcesat-2",
    "Resourcesat-2A",
    "Cartosat-1",
    "Cartosat-2",
    "Cartosat-2A",
    "Cartosat-2B",
    "Cartosat-3",
    "Oceansat-1",
    "Oceansat-2",
    "EOS-06 (Oceansat-3)",
    "Bhuvan Indian Geo-Platform",
]

