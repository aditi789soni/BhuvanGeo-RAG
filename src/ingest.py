"""
Ingestion Pipeline for BhuvanGeo-RAG.
Coordinates PDF extraction, chunking, embedding generation, and ChromaDB persistence.
Ensures idempotency (no duplicate chunk additions).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.config import DOCUMENTS_DIR, PROCESSED_DIR
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.fetch_docs import collect_isro_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(force_clear: bool = False) -> Dict[str, Any]:
    """
    Execute full ingestion:
    1. Ensure documents exist in data/documents
    2. Extract and chunk text with PyMuPDF
    3. Ingest into ChromaDB with sentence-transformers embeddings
    4. Save processed cache
    """
    logger.info("=== Starting BhuvanGeo-RAG Ingestion Pipeline ===")

    # Step 1: Document collection / verification
    manifest_file = DOCUMENTS_DIR / "documents_manifest.json"
    if not manifest_file.exists():
        logger.info("Manifest not found. Running document collection...")
        collect_isro_documents()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Step 2: Initialize vector store and processor
    vector_store = VectorStore()
    if force_clear:
        logger.warning("Force clear enabled. Resetting Chroma collection...")
        vector_store.clear()

    processor = PDFProcessor()
    all_chunks = []
    doc_reports = []

    # Process all available PDFs in data/documents
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in data/documents. Running collector...")
        collect_isro_documents()
        pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF document(s) for ingestion.")

    for pdf_path in pdf_files:
        # Find matching entry in manifest for URL and metadata
        matching_meta = next(
            (d for d in manifest.get("documents", []) if d["filename"] == pdf_path.name),
            None,
        )

        source_url = matching_meta["primary_url"] if matching_meta else ""
        extra_meta = {
            "title": matching_meta["title"] if matching_meta else pdf_path.stem,
            "category": matching_meta["category"] if matching_meta else "General",
        }

        chunks = processor.process_document(
            pdf_path=pdf_path,
            source_url=source_url,
            extra_metadata=extra_meta,
        )
        all_chunks.extend(chunks)

        doc_reports.append(
            {
                "filename": pdf_path.name,
                "chunks_count": len(chunks),
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 2),
            }
        )

    # Step 3: Ingest chunks into ChromaDB
    logger.info(f"Ingesting total {len(all_chunks)} chunks into ChromaDB...")
    newly_added = vector_store.add_chunks(all_chunks)

    # Step 4: Write cache of chunk metadata
    cache_path = PROCESSED_DIR / "chunks_summary.json"
    summary_data = {
        "total_chunks": len(all_chunks),
        "newly_added_chunks": newly_added,
        "documents": doc_reports,
        "collection_stats": vector_store.get_document_stats(),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("=== Ingestion Pipeline Completed Successfully ===")
    logger.info(f"Total Chunks: {len(all_chunks)} | Newly Inserted: {newly_added}")
    return summary_data


if __name__ == "__main__":
    import sys
    should_clear = "--reset" in sys.argv
    run_ingestion(force_clear=should_clear)

