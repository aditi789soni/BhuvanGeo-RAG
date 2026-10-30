"""
Vector Database Manager for BhuvanGeo-RAG.
Embeds document chunks using sentence-transformers and persists them in ChromaDB.
Provides idempotent ingestion and fast semantic similarity retrieval.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVAL,
)
from src.pdf_processor import PDFChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LocalEmbeddingFunction:
    """Wrapper around SentenceTransformer for ChromaDB compatibility."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            input_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class VectorStore:
    """ChromaDB-backed vector database with idempotent ingestion and similarity query."""

    def __init__(
        self,
        persist_directory: Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize local sentence-transformers embedding function
        self.embedder = LocalEmbeddingFunction(model_name=embedding_model_name)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "ISRO and Bhuvan Geospatial Mission Documents"},
        )
        logger.info(
            f"ChromaDB initialized at '{self.persist_directory}'. "
            f"Collection '{self.collection_name}' contains {self.collection.count()} chunks."
        )

    def add_chunks(self, chunks: List[PDFChunk], batch_size: int = 64) -> int:
        """
        Idempotently ingest chunks into ChromaDB.
        Skips chunks that already exist based on chunk_id.
        Returns the number of newly added chunks.
        """
        if not chunks:
            logger.info("No chunks provided to add.")
            return 0

        # Check existing chunk IDs in collection to avoid duplicates
        existing_ids = set()
        try:
            # Query existing IDs in batches or all
            existing_records = self.collection.get(ids=[c.chunk_id for c in chunks])
            if existing_records and "ids" in existing_records:
                existing_ids = set(existing_records["ids"])
        except Exception as e:
            logger.warning(f"Failed to check existing IDs: {e}")

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info(
                f"All {len(chunks)} chunks already exist in ChromaDB collection. Skipping ingestion."
            )
            return 0

        logger.info(
            f"Adding {len(new_chunks)} new chunks (skipped {len(chunks) - len(new_chunks)} duplicates)..."
        )

        # Ingest in batches
        total_added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.to_dict()["metadata"] for c in batch]

            # Generate normalized embeddings
            embeddings = self.embedder(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total_added += len(batch)

        logger.info(
            f"Successfully added {total_added} chunks. "
            f"Total collection size: {self.collection.count()} items."
        )
        return total_added

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RETRIEVAL,
        document_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks matching the query text.
        Returns a list of structured results with similarity scores.
        """
        if not query_text or not query_text.strip():
            return []

        if self.collection.count() == 0:
            logger.warning("Vector store is empty! No documents have been ingested yet.")
            return []

        # Embed query text
        query_embedding = self.embedder([query_text])[0]

        where_filter = None
        if document_filter:
            where_filter = {"document_name": document_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # Chroma returns L2 or cosine distance. Convert distance to a normalized similarity score (0.0 - 1.0)
            # For normalized cosine embeddings, cosine distance = 1 - cosine_similarity (range 0 to 2)
            similarity_score = max(0.0, 1.0 - (dist / 2.0))

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta,
                    "document_name": meta.get("document_name", "Unknown"),
                    "page_number": meta.get("page_number", 1),
                    "source_url": meta.get("source_url", ""),
                    "distance": float(dist),
                    "similarity_score": round(float(similarity_score), 4),
                }
            )

        return formatted_results

    def get_document_stats(self) -> Dict[str, Any]:
        """Return collection statistics and distinct ingested documents."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "documents": {}}

        data = self.collection.get(include=["metadatas"])
        doc_counts = {}
        for meta in data.get("metadatas", []):
            dname = meta.get("document_name", "Unknown")
            doc_counts[dname] = doc_counts.get(dname, 0) + 1

        return {
            "total_chunks": count,
            "documents": doc_counts,
        }

    def clear(self) -> None:
        """Reset the Chroma collection (useful for clean re-ingestion/testing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "ISRO and Bhuvan Geospatial Mission Documents"},
        )
        logger.info(f"Cleared collection '{self.collection_name}'.")

