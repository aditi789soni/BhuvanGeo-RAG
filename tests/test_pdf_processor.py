"""
Unit tests for the PDF Processing module.
Verifies text cleaning, page extraction, chunking, and metadata preservation.
"""

import pytest
from pathlib import Path
import pymupdf as fitz
from src.pdf_processor import PDFProcessor, PDFChunk


def test_clean_text_hyphenation():
    """Verify that hyphenated line breaks are stitched properly."""
    raw = "The spatial resolu-\ntion of LISS-4 is 5.8 meters."
    cleaned = PDFProcessor.clean_text(raw)
    assert "resolution" in cleaned
    assert "resolu-" not in cleaned


def test_clean_text_whitespace():
    """Verify whitespace and multiple newline normalization."""
    raw = "Multiple   spaces \t\t and\r\n\r\n\r\nparagraphs."
    cleaned = PDFProcessor.clean_text(raw)
    assert "   " not in cleaned
    assert "\r" not in cleaned
    assert "\n\n" in cleaned


def test_chunk_page_text_small():
    """Verify that a small text segment is returned as a single chunk."""
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    text = "Cartosat-1 was launched by ISRO for stereoscopic terrain mapping."
    chunks = processor.chunk_page_text(
        text=text, doc_name="Cartosat-1.pdf", page_num=1, total_pages=5
    )
    assert len(chunks) == 1
    assert chunks[0].document_name == "Cartosat-1.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].total_pages == 5
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "Cartosat-1_p1_c0"


def test_chunk_page_text_sliding_window():
    """Verify sliding window chunking with overlap for longer text."""
    processor = PDFProcessor(chunk_size=150, chunk_overlap=30)
    long_text = (
        "Linear Imaging Self-Scanning Sensor-4 (LISS-IV) is an advanced camera. "
        "It provides 5.8 m spatial resolution across VNIR bands. "
        "Resourcesat-2 enhanced the swath width to 70 km in multispectral mode. "
        "It supports both mono and multispectral operational imaging modes."
    )
    chunks = processor.chunk_page_text(
        text=long_text, doc_name="Resourcesat-2.pdf", page_num=2, total_pages=10
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.document_name == "Resourcesat-2.pdf"
        assert c.page_number == 2
        assert len(c.text) >= 30
        assert c.chunk_id.startswith("Resourcesat-2_p2_c")


def test_synthetic_pdf_extraction(tmp_path: Path):
    """Create a temporary synthetic PDF and verify extraction with PyMuPDF."""
    test_pdf_file = tmp_path / "test_isro_doc.pdf"

    # Create a 2-page PDF
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "ISRO Resourcesat-2 carries three payloads: LISS-4, LISS-3, and AWiFS.",
        fontsize=12,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Cartosat-1 carries two panchromatic cameras: Fore and Aft for stereo DEM.",
        fontsize=12,
    )
    doc.save(str(test_pdf_file))
    doc.close()

    processor = PDFProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.process_document(test_pdf_file, source_url="https://isro.gov.in/test")

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Resourcesat-2" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "Cartosat-1" in chunks[1].text
    assert chunks[0].metadata["source_url"] == "https://isro.gov.in/test"
