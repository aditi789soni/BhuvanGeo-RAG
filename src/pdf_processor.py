"""
PDF Processing Engine for BhuvanGeo-RAG.
Uses PyMuPDF (fitz) to extract text, preserve page numbers and document metadata,
and segment documents into semantically coherent overlapping chunks.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf as fitz

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PDFChunk:
    """Represents an extracted and chunked segment of a PDF document."""

    def __init__(
        self,
        chunk_id: str,
        document_name: str,
        page_number: int,
        total_pages: int,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.document_name = document_name
        self.page_number = page_number
        self.total_pages = total_pages
        self.text = text
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "text": self.text,
            "metadata": {
                **self.metadata,
                "document_name": self.document_name,
                "page_number": self.page_number,
                "total_pages": self.total_pages,
                "char_count": len(self.text),
            },
        }


class PDFProcessor:
    """Extracts, cleans, and chunks text from PDF files using PyMuPDF."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Clean and normalize raw extracted PDF text.
        - Fix hyphenated line breaks (e.g., 'geo-\nspatial' -> 'geospatial')
        - Replace unusual unicode spaces and multiple whitespaces
        - Normalize line breaks while retaining paragraph structure
        """
        if not raw_text:
            return ""

        # Remove hyphenation at line endings: word-\nword -> wordword
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)

        # Replace carriage returns and excessive whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple spaces or tabs into a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3 or more newlines into 2 (paragraph separator)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract page-by-page text from a PDF file using PyMuPDF.
        Handles missing/scanned pages gracefully.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        extracted_pages = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path.name}: {e}")
            raise

        total_pages = len(doc)
        logger.info(f"Processing '{pdf_path.name}' with {total_pages} pages.")

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]

            try:
                # Extract plain text
                raw_text = page.get_text("text")
                cleaned_text = self.clean_text(raw_text)

                # Check for scanned/empty pages
                if len(cleaned_text) < 25:
                    logger.warning(
                        f"Page {page_num}/{total_pages} in '{pdf_path.name}' has little to no text "
                        f"({len(cleaned_text)} chars). Might be scanned or purely graphical."
                    )
                    is_scanned_or_empty = True
                else:
                    is_scanned_or_empty = False

                extracted_pages.append(
                    {
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "text": cleaned_text,
                        "is_scanned_or_empty": is_scanned_or_empty,
                    }
                )
            except Exception as page_err:
                logger.warning(f"Error reading page {page_num} of {pdf_path.name}: {page_err}")
                extracted_pages.append(
                    {
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "text": "",
                        "is_scanned_or_empty": True,
                    }
                )

        doc.close()
        return extracted_pages

    def chunk_page_text(
        self,
        text: str,
        doc_name: str,
        page_num: int,
        total_pages: int,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[PDFChunk]:
        """
        Split a single page's text into overlapping chunks.
        Splits by natural paragraph/sentence boundaries where possible.
        """
        if not text or len(text.strip()) < 20:
            return []

        chunks: List[PDFChunk] = []
        doc_clean_name = re.sub(r"[^\w\-_\.]", "_", doc_name).replace(".pdf", "")

        # If text is smaller than chunk size, return it as a single chunk
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc_clean_name}_p{page_num}_c0"
            chunks.append(
                PDFChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    page_number=page_num,
                    total_pages=total_pages,
                    text=text,
                    metadata=base_metadata or {},
                )
            )
            return chunks

        # Sliding window chunking with paragraph/sentence boundary preference
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end of text, find the best breaking point near the end
            if end < text_len:
                search_start = max(start + self.chunk_size // 2, end - self.chunk_overlap)
                para_break = text.rfind("\n\n", search_start, end)
                if para_break != -1 and para_break > search_start:
                    end = para_break + 2
                else:
                    sentence_break = text.rfind(". ", search_start, end)
                    if sentence_break != -1 and sentence_break > search_start:
                        end = sentence_break + 2
                    else:
                        space_break = text.rfind(" ", search_start, end)
                        if space_break != -1 and space_break > search_start:
                            end = space_break + 1

            chunk_text = text[start:end].strip()
            if len(chunk_text) >= 40:
                chunk_id = f"{doc_clean_name}_p{page_num}_c{chunk_idx}"
                chunks.append(
                    PDFChunk(
                        chunk_id=chunk_id,
                        document_name=doc_name,
                        page_number=page_num,
                        total_pages=total_pages,
                        text=chunk_text,
                        metadata=base_metadata or {},
                    )
                )
                chunk_idx += 1

            # Advance by at least chunk_size - chunk_overlap
            min_advance = max(self.chunk_size - self.chunk_overlap, 100)
            actual_advance = end - start - self.chunk_overlap
            start += max(actual_advance, min_advance)

        return chunks

    def process_document(
        self,
        pdf_path: Path,
        source_url: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[PDFChunk]:
        """
        Process a full PDF document: extract pages and generate enriched chunks.
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.name

        base_metadata = extra_metadata or {}
        if source_url:
            base_metadata["source_url"] = source_url

        pages = self.extract_pages(pdf_path)
        all_chunks: List[PDFChunk] = []

        for page_data in pages:
            if page_data["is_scanned_or_empty"]:
                continue

            page_chunks = self.chunk_page_text(
                text=page_data["text"],
                doc_name=doc_name,
                page_num=page_data["page_number"],
                total_pages=page_data["total_pages"],
                base_metadata=base_metadata,
            )
            all_chunks.extend(page_chunks)

        logger.info(
            f"Extracted {len(all_chunks)} chunks from '{doc_name}' ({len(pages)} pages total)."
        )
        return all_chunks

