"""Tests for research intelligence modules, PRISMA counts, network, verifier & orchestrator."""

from __future__ import annotations

from agents.master_orchestrator import MasterResearchOrchestrator
from core.llm_client import GeminiClient
from core.project_manager import create_empty_project
from core.schemas import ScholarlyWork
from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from research.citation_network import analyze_graph_metrics, build_citation_graph
from research.citation_verifier import verify_claim_against_evidence
from research.consistency_checker import run_consistency_check
from research.paper_discovery import deduplicate_works
from research.systematic_review import compute_prisma_counts
from tests.test_document_engine import _make_sample_pdf_bytes


def test_prisma_counts_from_real_screening_decisions() -> None:
    works = [
        ScholarlyWork(work_id="w1", title="Paper One", year=2022, screening_status="included", full_text_assessed=True),
        ScholarlyWork(work_id="w2", title="Paper One", year=2022),  # duplicate
        ScholarlyWork(work_id="w3", title="Paper Two", year=2015, screening_status="excluded", exclusion_reason="Too old"),
        ScholarlyWork(work_id="w4", title="Paper Three", year=2023, screening_status="excluded", full_text_assessed=True, exclusion_reason="No dataset"),
        ScholarlyWork(work_id="w5", title="Paper Four", year=2024, screening_status="needs_review"),
    ]
    deduplicate_works(works)
    prisma = compute_prisma_counts(works)
    assert prisma.records_identified == 5
    assert prisma.duplicates_removed == 1
    assert prisma.records_screened == 4
    assert prisma.records_excluded == 1
    assert prisma.full_text_assessed == 2
    assert prisma.full_text_excluded == 1
    assert prisma.studies_included == 1
    assert prisma.needs_review == 1


def test_citation_verifier_never_verifies_by_doi_alone() -> None:
    vs = SessionVectorStore("test_ver")
    # Even if a work has a valid DOI, with empty abstract/no PDF passages it MUST be Unable to verify
    work_with_doi_only = ScholarlyWork(
        work_id="w_doi",
        title="Unverified Paper",
        doi="10.1000/xyz123",
        abstract="",
    )
    res = verify_claim_against_evidence(
        claim_text="Our algorithm achieves 99% accuracy [1].",
        cited_source="[1]",
        section_name="Results",
        vector_store=vs,
        discovered_works=[work_with_doi_only],
    )
    assert res.support_status == "Unable to verify"


def test_orchestrator_citation_network_and_consistency() -> None:
    proj = create_empty_project(title="RAG Verification")
    parsed = process_pdf_bytes(_make_sample_pdf_bytes(), "rag.pdf")
    proj.parsed_papers.append(parsed)
    proj.discovered_works.append(
        ScholarlyWork(
            work_id="oa_1",
            source="openalex",
            title="Transformer Retrieval Methods",
            authors=["Alice Smith"],
            year=2023,
            fields_of_study=["Natural Language Processing"],
            abstract="Transformer retrieval improves scientific question answering.",
        )
    )
    vs = SessionVectorStore(proj.project_id)
    vs.build_from_chunks(parsed.chunks)
    llm = GeminiClient(api_key="")  # offline mode

    orch = MasterResearchOrchestrator(proj, vs, llm)
    out_pi = orch.run("paper_intelligence", query=parsed.title, params={"paper": parsed})
    assert out_pi["result"].paper_id == parsed.paper_id

    proj.paper_analyses[parsed.paper_id] = out_pi["result"]
    out_gaps = orch.run("gap_detection")
    assert len(out_gaps["result"]) >= 1

    G = build_citation_graph(proj.discovered_works, proj.parsed_papers, proj.paper_analyses)
    metrics = analyze_graph_metrics(G)
    assert metrics["nodes"] > 0
    assert metrics["edges"] > 0

    rep = run_consistency_check(proj)
    assert 0.0 <= rep.overall_score <= 100.0
