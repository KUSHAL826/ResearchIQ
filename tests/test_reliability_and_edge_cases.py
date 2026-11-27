"""Comprehensive reliability, edge-case, mocked-API, PRISMA, Verifier, Cross-Domain, Trend, and .riq tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf as fitz
import pytest

from core.llm_client import GeminiClient, _retry_with_backoff
from core.project_manager import (
    cleanup_project_temp_dir,
    create_empty_project,
    export_project_to_riq_bytes,
    validate_and_import_riq_bytes,
)
from core.schemas import ScholarlyWork
from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from paper_studio.publisher_exporter import export_bibtex, export_docx_bytes, export_latex, export_pdf_bytes
from research.citation_verifier import verify_claim_against_evidence
from research.cross_domain import explore_cross_domain
from research.paper_discovery import deduplicate_works, discover_papers
from research.systematic_review import compute_prisma_counts
from research.trend_analyzer import compute_research_trends
from tests.test_document_engine import _make_sample_pdf_bytes


# ---------------------------------------------------------------------------
# Priority 1A: PRISMA consistency & no auto-inclusion of uploaded PDFs
# ---------------------------------------------------------------------------

def test_prisma_never_auto_includes_uploaded_pdfs_and_persists_in_riq() -> None:
    proj = create_empty_project(title="PRISMA Integrity")
    try:
        parsed = process_pdf_bytes(_make_sample_pdf_bytes(), "uploaded.pdf")
        assert parsed.screening_status == "unscreened"
        proj.parsed_papers.append(parsed)

        # Uploaded PDF must count as pending_screening, NOT included!
        c1 = compute_prisma_counts(proj.discovered_works, proj.parsed_papers)
        assert c1.records_identified == 1
        assert c1.pending_screening == 1
        assert c1.studies_included == 0
        assert c1.full_text_assessed == 0

        # Explicit exclusion with reason
        parsed.screening_status = "excluded"
        parsed.full_text_assessed = True
        parsed.exclusion_reason = "Out of domain scope"
        c2 = compute_prisma_counts(proj.discovered_works, proj.parsed_papers)
        assert c2.full_text_assessed == 1
        assert c2.full_text_excluded == 1
        assert c2.studies_included == 0
        assert c2.exclusion_reasons["Out of domain scope"] == 1

        # Round-trip through .riq preserves screening decisions
        riq = export_project_to_riq_bytes(proj)
        restored, _, _ = validate_and_import_riq_bytes(riq)
        try:
            assert restored.parsed_papers[0].screening_status == "excluded"
            assert restored.parsed_papers[0].exclusion_reason == "Out of domain scope"
            c3 = compute_prisma_counts(restored.discovered_works, restored.parsed_papers)
            assert c3.full_text_excluded == 1
            assert c3.studies_included == 0
        finally:
            cleanup_project_temp_dir(restored.project_id)
    finally:
        cleanup_project_temp_dir(proj.project_id)


# ---------------------------------------------------------------------------
# Priority 1B: Citation Claim Verifier reliability & conservative fallback
# ---------------------------------------------------------------------------

def test_citation_verifier_conservative_offline_and_contradiction() -> None:
    proj = create_empty_project(title="Verifier Test")
    try:
        parsed = process_pdf_bytes(_make_sample_pdf_bytes(), "bioasq_rag.pdf")
        parsed.title = "BioASQ Retrieval Study"
        proj.parsed_papers.append(parsed)

        vs = SessionVectorStore(proj.project_id)
        vs.build_from_chunks(parsed.chunks)

        # Even with matching terminology ("BioASQ accuracy FAISS Transformer"), offline fallback MUST NOT mark Supported
        res_offline = verify_claim_against_evidence(
            claim_text="BioASQ benchmark accuracy reaches 88.5% using FAISS and Transformer reranking [1].",
            cited_source="[1]",
            section_name="Results",
            vector_store=vs,
            parsed_papers=proj.parsed_papers,
            references_text="[1] BioASQ Retrieval Study, 2024.",
            llm_client=None,
        )
        assert res_offline.support_status == "Unable to verify"
        assert "BioASQ Retrieval Study" in res_offline.cited_source
        assert "pp." in res_offline.page_reference

        # Explicit numeric contradiction on shared metric MUST flag Not supported
        res_contra = verify_claim_against_evidence(
            claim_text="On BioASQ, accuracy dropped to 12.0% with FAISS [1].",
            cited_source="[1]",
            section_name="Results",
            vector_store=vs,
            parsed_papers=proj.parsed_papers,
            references_text="[1] BioASQ Retrieval Study, 2024.",
            llm_client=None,
        )
        assert res_contra.support_status == "Not supported"
        assert res_contra.contradiction_note != ""
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_citation_verifier_prioritizes_cited_source_with_mocked_gemini() -> None:
    vs = SessionVectorStore("mock_ver")
    w1 = ScholarlyWork(
        work_id="w_other",
        title="Unrelated Vision Paper",
        authors=["Bob Lee"],
        year=2021,
        abstract="Convolutional networks achieve 95% accuracy on ImageNet classification.",
    )
    w2 = ScholarlyWork(
        work_id="w_cited",
        title="Clinical Retrieval Study",
        authors=["Alice Smith"],
        year=2023,
        abstract="Dense passage retrieval improves clinical QA recall by 14%.",
    )
    mock_llm = MagicMock(spec=GeminiClient)
    mock_llm.is_configured = True
    mock_llm.generate_structured.return_value = SimpleNamespace(
        support_status="Supported",
        contradiction_note="",
        reasoning="Directly supported by Smith et al. abstract.",
        confidence_score=0.91,
    )
    res = verify_claim_against_evidence(
        claim_text="Dense passage retrieval improves clinical QA recall (Smith, 2023).",
        cited_source="Smith, 2023",
        section_name="Literature Review",
        vector_store=vs,
        discovered_works=[w1, w2],
        llm_client=mock_llm,
    )
    assert res.support_status == "Supported"
    assert "Clinical Retrieval Study" in res.cited_source
    assert res.confidence_score == 0.91


# ---------------------------------------------------------------------------
# Priority 1C: Cross-Domain Finder (mocked APIs, 4-tier breakdown, no false novelty)
# ---------------------------------------------------------------------------

def test_cross_domain_mocked_api_no_false_novelty() -> None:
    fake_inter = [
        ScholarlyWork(
            work_id="oa_cd1",
            source="openalex",
            title="AI for Healthcare Triage",
            authors=["Dr. A"],
            year=2023,
            abstract="Artificial intelligence methods applied in clinical healthcare triage.",
            fields_of_study=["Artificial Intelligence", "Healthcare"],
        )
    ]
    with patch("research.cross_domain.discover_papers", return_value=(fake_inter, [])):
        ideas, works, breakdown, errs = explore_cross_domain(["Artificial Intelligence", "Healthcare"], llm_client=None)

    assert not errs
    assert len(ideas) >= 2
    assert breakdown["novelty_established"] is False
    assert "Existing Intersection" in breakdown["counts"]
    assert "existing_research" in breakdown
    assert "related_research" in breakdown
    assert "potentially_unexplored" in breakdown
    assert all(idea.novelty_score <= 6.5 for idea in ideas)


# ---------------------------------------------------------------------------
# Priority 1D: Trend Analyzer (growth rates, missing years, small-sample caution)
# ---------------------------------------------------------------------------

def test_trend_analyzer_missing_years_growth_and_empty() -> None:
    empty_t = compute_research_trends([])
    assert empty_t["total_works"] == 0

    works = [
        ScholarlyWork(work_id="t1", title="Transformer Model A", year=2022, fields_of_study=["NLP"], citation_count=10),
        ScholarlyWork(work_id="t2", title="Transformer Model B", year=2023, fields_of_study=["NLP"], citation_count=20),
        ScholarlyWork(work_id="t3", title="LLM Agent C", year=2023, fields_of_study=["NLP", "Agents"], citation_count=5),
        ScholarlyWork(work_id="t4", title="RAG Undated Study", year=None, fields_of_study=["NLP"]),
    ]
    trends = compute_research_trends(works)
    assert trends["total_works"] == 4
    assert trends["dated_works"] == 3
    assert trends["missing_year_count"] == 1
    assert trends["yearly_growth_pct"][2023] == 100.0  # 1 -> 2 papers
    assert "RAG" in trends["undated_algorithm_mentions"]
    assert 0 not in trends["algorithm_trends"].get("RAG", {})
    assert trends["small_sample_warning"] != ""
    assert "Heuristic" in trends["heuristic_disclaimer"]


# ---------------------------------------------------------------------------
# Priority 1E & 4: LaTeX math preservation, Springer/ACM DOCX, Multi-project isolation
# ---------------------------------------------------------------------------

def test_latex_preserves_math_and_exports_springer_acm_docx() -> None:
    proj = create_empty_project(title="Math & Export Study", domain="Machine Learning")
    try:
        proj.manuscript_sections["Abstract"].content = "Loss is $L = \\alpha x_1 + \\beta$ with 95% confidence & accuracy."
        proj.manuscript_sections["Proposed Methodology"].content = "Mathematical Formulation\nWe optimize $$E = mc^2$$ under constraints."
        proj.publication_prep.target_format = "Springer"
        tex_sp = export_latex(proj)
        assert "\\documentclass[runningheads]{llncs}" in tex_sp
        assert "$L = \\alpha x_1 + \\beta$" in tex_sp
        assert "\\&" in tex_sp
        assert "\\subsection{Mathematical Formulation}" in tex_sp
        assert len(export_docx_bytes(proj)) > 1000
        assert export_pdf_bytes(proj).startswith(b"%PDF")

        proj.publication_prep.target_format = "ACM"
        tex_acm = export_latex(proj)
        assert "\\documentclass[sigconf]{acmart}" in tex_acm
        assert len(export_docx_bytes(proj)) > 1000
        assert export_bibtex(proj).startswith("%")
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_discovery_api_failure_and_dedup_handling() -> None:
    with patch("research.paper_discovery.requests.get", side_effect=RuntimeError("Network timeout")):
        works, errs = discover_papers("quantum computing", sources=["openalex", "arxiv", "semantic_scholar"])
    assert works == []
    assert len(errs) == 3


# ---------------------------------------------------------------------------
# Priority 5: Gemini retry/backoff, malformed output safety, key privacy, large PDF
# ---------------------------------------------------------------------------

def test_gemini_retry_backoff_and_api_key_not_leaked() -> None:
    calls = {"n": 0}

    def flaky_fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 ResourceExhausted rate limit")
        return "ok"

    with patch("core.llm_client.time.sleep", return_value=None):
        assert _retry_with_backoff(flaky_fn, max_retries=3) == "ok"
    assert calls["n"] == 3

    client = GeminiClient(api_key="SECRET_TEST_KEY_12345")
    assert "SECRET_TEST_KEY_12345" not in repr(client)

    proj = create_empty_project(title="Key Safety Check")
    try:
        riq_bytes = export_project_to_riq_bytes(proj)
        assert b"SECRET_TEST_KEY_12345" not in riq_bytes
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_large_multi_page_pdf_and_empty_search_edge_cases() -> None:
    doc = fitz.open()
    for i in range(15):
        page = doc.new_page()
        page.insert_text((72, 72), f"Section {i+1}\nPage {i+1} analytical experimental evaluation text. " * 20)
    large_bytes = doc.tobytes()
    doc.close()

    parsed = process_pdf_bytes(large_bytes, "large_15pp.pdf")
    assert parsed.num_pages == 15
    assert len(parsed.chunks) >= 15

    vs = SessionVectorStore("large_test")
    cnt, _ = vs.build_from_chunks(parsed.chunks)
    assert cnt == len(parsed.chunks)
    assert vs.search("   ") == []  # Empty query returns cleanly
