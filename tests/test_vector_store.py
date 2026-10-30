"""
Unit tests for the Vector Store module.
Verifies ChromaDB collection initialization, idempotent chunk ingestion,
similarity retrieval, and document filtering.
"""

import pytest
from pathlib import Path
from src.vector_store import VectorStore
from src.pdf_processor import PDFChunk


@pytest.fixture
def temp_vector_store(tmp_path: Path):
    """Create a temporary VectorStore instance using a temporary directory."""
    vs = VectorStore(
        persist_directory=tmp_path / "test_chroma",
        collection_name="test_isro_collection",
    )
    yield vs


def test_vector_store_initialization(temp_vector_store: VectorStore):
    """Verify that vector store initializes with empty collection."""
    assert temp_vector_store.collection.count() == 0
    stats = temp_vector_store.get_document_stats()
    assert stats["total_chunks"] == 0


def test_idempotent_ingestion(temp_vector_store: VectorStore):
    """Verify that inserting the same chunks multiple times does not create duplicates."""
    chunks = [
        PDFChunk(
            chunk_id="doc1_p1_c0",
            document_name="Resourcesat-2.pdf",
            page_number=1,
            total_pages=5,
            text="LISS-IV camera operates in VNIR bands with 5.8 meter resolution.",
        ),
        PDFChunk(
            chunk_id="doc1_p2_c0",
            document_name="Resourcesat-2.pdf",
            page_number=2,
            total_pages=5,
            text="AWiFS sensor provides 56 meter resolution and 740 km swath.",
        ),
    ]

    # First ingestion
    added_first = temp_vector_store.add_chunks(chunks)
    assert added_first == 2
    assert temp_vector_store.collection.count() == 2

    # Second ingestion of same chunks (should skip all)
    added_second = temp_vector_store.add_chunks(chunks)
    assert added_second == 0
    assert temp_vector_store.collection.count() == 2


def test_similarity_query(temp_vector_store: VectorStore):
    """Verify semantic retrieval and similarity scoring."""
    chunks = [
        PDFChunk(
            chunk_id="c_liss4",
            document_name="Resourcesat-2.pdf",
            page_number=1,
            total_pages=5,
            text="The LISS-IV payload provides 5.8 m spatial resolution across three spectral bands B2, B3, B4.",
        ),
        PDFChunk(
            chunk_id="c_cartosat",
            document_name="Cartosat-1.pdf",
            page_number=1,
            total_pages=4,
            text="Cartosat-1 carries Fore and Aft panchromatic cameras for stereoscopic cartographic DEM.",
        ),
    ]
    temp_vector_store.add_chunks(chunks)

    # Query about LISS-4
    results = temp_vector_store.query("What is the resolution of LISS-IV sensor?", top_k=2)
    assert len(results) >= 1
    top_result = results[0]
    assert top_result["chunk_id"] == "c_liss4"
    assert top_result["similarity_score"] > 0.4
    assert top_result["document_name"] == "Resourcesat-2.pdf"

    # Query about Cartosat stereoscopy
    results_carto = temp_vector_store.query("Cartosat stereo cameras Fore Aft DEM", top_k=2)
    assert results_carto[0]["chunk_id"] == "c_cartosat"


def test_document_filter(temp_vector_store: VectorStore):
    """Verify filtering by specific document name."""
    chunks = [
        PDFChunk(
            chunk_id="r2_chunk",
            document_name="Resourcesat-2.pdf",
            page_number=1,
            total_pages=5,
            text="Earth observation multispectral imaging.",
        ),
        PDFChunk(
            chunk_id="c1_chunk",
            document_name="Cartosat-1.pdf",
            page_number=1,
            total_pages=5,
            text="Earth observation panchromatic stereo imaging.",
        ),
    ]
    temp_vector_store.add_chunks(chunks)

    filtered_results = temp_vector_store.query(
        "imaging", top_k=5, document_filter="Cartosat-1.pdf"
    )
    assert len(filtered_results) == 1
    assert filtered_results[0]["document_name"] == "Cartosat-1.pdf"

