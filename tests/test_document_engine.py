"""Tests for PDF validation, extraction, section parsing, chunking, and session FAISS."""

from __future__ import annotations

import pymupdf as fitz
import pytest

from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from document_engine.pdf_extractor import validate_pdf_bytes


def _make_sample_pdf_bytes() -> bytes:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(
        (72, 72),
        "Robust Retrieval-Augmented Generation for Scientific QA\n"
        "Abstract\n"
        "We propose a section-aware retrieval architecture for scientific papers on the BioASQ dataset.\n"
        "1. Introduction\n"
        "Large language models often hallucinate citations without verifiable page grounding.\n"
        "2. Methodology\n"
        "We combine FAISS dense embeddings with Transformer reranking.\n",
    )
    p2 = doc.new_page()
    p2.insert_text(
        (72, 72),
        "3. Experiments\n"
        "Evaluated on BioASQ benchmark with accuracy 88.5% and F1-score 86.2%.\n"
        "4. Results\n"
        "Our method outperforms BM25 baselines.\n"
        "5. Conclusion\n"
        "A key limitation is sensitivity to scanned OCR noise and single-domain benchmarks.\n",
    )
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_validation_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError):
        validate_pdf_bytes(b"not a pdf", "bad.pdf")


def test_process_pdf_and_vector_search() -> None:
    pdf_bytes = _make_sample_pdf_bytes()
    stages: list[str] = []
    parsed = process_pdf_bytes(pdf_bytes, "sample_rag.pdf", progress_cb=stages.append)

    assert parsed.num_pages == 2
    assert "Document uploaded" in stages
    assert "Text extracted" in stages
    assert "Sections detected" in stages
    assert len(parsed.sections) >= 4
    assert len(parsed.chunks) >= 2

    vs = SessionVectorStore(project_id="test_proj")
    count, mode = vs.build_from_chunks(parsed.chunks, llm_client=None)
    assert count == len(parsed.chunks)
    assert mode == "lexical_fallback"

    hits = vs.search("BioASQ accuracy FAISS Transformer", top_k=2)
    assert len(hits) >= 1
    assert hits[0][0].paper_id == parsed.paper_id
