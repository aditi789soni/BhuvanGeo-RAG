"""
Unit tests for the RAG Question Answering Engine.
Verifies grounded response synthesis, citation verification,
and anti-hallucination guardrails for unsupported queries.
"""

import pytest
from pathlib import Path
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.pdf_processor import PDFChunk


@pytest.fixture
def populated_rag_engine(tmp_path: Path):
    """Fixture providing a RAGEngine populated with test ISRO sensor chunks."""
    vs = VectorStore(
        persist_directory=tmp_path / "rag_chroma",
        collection_name="test_rag_collection",
    )

    sample_chunks = [
        PDFChunk(
            chunk_id="chunk_liss4_specs",
            document_name="Resourcesat-2_User_Handbook.pdf",
            page_number=14,
            total_pages=45,
            text=(
                "The LISS-IV camera operates in the VNIR region with a spatial resolution of 5.8 m at nadir. "
                "In multispectral mode, the swath is 70 km across three bands: Green (B2), Red (B3), and NIR (B4)."
            ),
        ),
        PDFChunk(
            chunk_id="chunk_awifs_specs",
            document_name="Resourcesat-2_User_Handbook.pdf",
            page_number=22,
            total_pages=45,
            text=(
                "AWiFS sensor has two camera heads, AWiFS-A and AWiFS-B, providing a combined swath of 740 km "
                "with a spatial resolution of 56 meters and a 5-day revisit cycle."
            ),
        ),
        PDFChunk(
            chunk_id="chunk_carto1_specs",
            document_name="Cartosat-1_Brochure.pdf",
            page_number=3,
            total_pages=12,
            text=(
                "Cartosat-1 carries two panchromatic cameras, Fore (+26 deg) and Aft (-5 deg), "
                "providing 2.5 m spatial resolution and stereoscopic pairs for CartoDEM generation."
            ),
        ),
    ]

    vs.add_chunks(sample_chunks)
    return RAGEngine(vector_store=vs)


def test_rag_supported_question(populated_rag_engine: RAGEngine):
    """Verify that a supported question returns grounded facts and citations."""
    query = "What is the spatial resolution and swath width of LISS-IV on Resourcesat-2?"
    response = populated_rag_engine.answer_question(query=query)

    assert response["is_grounded"] is True
    assert len(response["citations"]) >= 1
    assert "5.8" in response["answer"]
    assert "70" in response["answer"]

    # Verify citation fields
    top_citation = response["citations"][0]
    assert top_citation["document"] == "Resourcesat-2_User_Handbook.pdf"
    assert top_citation["page"] == 14
    assert len(top_citation["excerpt"]) > 0


def test_rag_unsupported_question(populated_rag_engine: RAGEngine):
    """
    Anti-hallucination test:
    Verify that an unanswerable / unrepresented question returns an explicit insufficiency notice.
    """
    unsupported_query = "What is the mass of the Pragyan rover on Chandrayaan-3 and its battery capacity?"
    response = populated_rag_engine.answer_question(query=unsupported_query)

    # Should report insufficient evidence rather than hallucinating specifications
    assert "Insufficient evidence" in response["answer"] or response["is_grounded"] is False


def test_rag_citation_integrity(populated_rag_engine: RAGEngine):
    """Verify that every citation provides exact page numbers and document names."""
    query = "How does Cartosat-1 generate stereo DEM data?"
    response = populated_rag_engine.answer_question(query=query)

    assert response["is_grounded"] is True
    assert len(response["citations"]) > 0

    carto_cit = next(
        (c for c in response["citations"] if "Cartosat-1" in c["document"]), None
    )
    assert carto_cit is not None
    assert carto_cit["page"] == 3
    assert "Fore" in carto_cit["excerpt"] or "Aft" in carto_cit["excerpt"]

