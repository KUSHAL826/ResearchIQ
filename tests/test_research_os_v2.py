"""Tests for RESEARCHIQ 2.0 Intelligence Layer, backward-compatible .riq serialization, and UI navigation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pandas as pd
from streamlit.testing.v1 import AppTest

from core.project_manager import (
    cleanup_project_temp_dir,
    create_empty_project,
    export_project_to_riq_bytes,
    validate_and_import_riq_bytes,
)
from core.schemas import ResearchGap, ScholarlyWork
from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from paper_studio.results_analyzer import register_uploaded_dataset
from research.paper_intelligence import analyze_paper_intelligence
from research_os.evidence_intel import collect_project_evidence
from research_os.experiment_engine import (
    compute_roc_and_pr_curves,
    compute_statistical_summary,
    explain_actual_results,
    extract_error_analysis_rows,
    sync_default_experiment_records,
)
from research_os.gap_validator import validate_research_gaps
from research_os.knowledge_graph import (
    build_unified_knowledge_graph,
    render_unified_kg_pyvis,
    summarize_unified_graph,
)
from research_os.landscape_intel import analyze_research_contradictions, map_research_landscape
from research_os.manuscript_intel import (
    analyze_manuscript_completeness,
    analyze_writing_quality,
    check_research_reproducibility,
    map_claims_to_evidence,
    run_academic_consistency_intelligence,
)
from research_os.methodology_architect import recommend_methodologies
from research_os.problem_architect import generate_problem_candidates, select_problem_candidate
from research_os.publication_intel import build_publication_suitability_report
from research_os.workflow_assistant import ask_project_workflow_assistant
from tests.test_document_engine import _make_sample_pdf_bytes


def test_research_os_v2_full_pipeline_and_riq_roundtrip() -> None:
    proj = create_empty_project(title="RESEARCHIQ 2.0 Study", domain="Artificial Intelligence")
    try:
        pdf_bytes = _make_sample_pdf_bytes()
        p1 = process_pdf_bytes(pdf_bytes, "rag_1.pdf")
        p2 = process_pdf_bytes(pdf_bytes, "rag_2.pdf")
        p2.title = "Cross-Dataset Clinical QA Benchmark"
        proj.parsed_papers = [p1, p2]
        a1 = analyze_paper_intelligence(p1)
        a2 = analyze_paper_intelligence(p2)
        # Give a2 a distinct dataset so contradiction intel checks experimental context
        a2.datasets = []
        proj.paper_analyses = {p1.paper_id: a1, p2.paper_id: a2}

        proj.discovered_works = [
            ScholarlyWork(
                work_id="oa_v2_1",
                source="openalex",
                title="Transformer RAG for Scientific QA",
                authors=["Alice Researcher"],
                year=2023,
                fields_of_study=["Natural Language Processing", "Information Retrieval"],
                abstract="Transformer RAG outperforms BM25 on BioASQ benchmark.",
                screening_status="included",
                full_text_assessed=True,
            )
        ]
        proj.research_gaps = [
            ResearchGap(
                title="Single-Domain Evaluation Bottleneck",
                description="Evaluation is limited to BioASQ.",
                supporting_publication=p1.title,
                paper_id=p1.paper_id,
                page_reference="p. 2",
                classification="explicit",
                evidence_quote="sensitivity to scanned OCR noise and single-domain benchmarks",
            )
        ]

        vs = SessionVectorStore(proj.project_id)
        vs.build_from_chunks([c for p in proj.parsed_papers for c in p.chunks])

        # 1. Unified Knowledge Graph
        G = build_unified_knowledge_graph(proj)
        summary = summarize_unified_graph(G)
        assert summary["nodes"] >= 5
        assert summary["edges"] >= 4
        kg_html = render_unified_kg_pyvis(G)
        assert "<html" in kg_html.lower()

        # 2. Evidence Intelligence
        ev_items = collect_project_evidence(proj, vector_store=vs, query="BioASQ FAISS")
        assert len(ev_items) >= 2
        assert any(e.page_number != "N/A" for e in ev_items)

        # 3. Landscape Mapping & Context-Aware Contradiction Intelligence
        ls = map_research_landscape(proj, year_min=2020, year_max=2025)
        assert ls["publication_activity"]
        contras = analyze_research_contradictions(proj)
        assert isinstance(contras, list)

        # 4. Gap Validation Engine
        g_reps = validate_research_gaps(proj, vector_store=vs)
        assert len(g_reps) == 1
        assert g_reps[0].uncertainties

        # 5. Problem Architect
        cands = generate_problem_candidates("Artificial Intelligence", "Clinical RAG", proj)
        assert len(cands) == 3
        select_problem_candidate(proj, cands[0])
        assert proj.selected_topic == cands[0].title

        # 6. Methodology Architect
        m_recs = recommend_methodologies(proj)
        assert len(m_recs) >= 2

        # 7. Experiment Engine (ROC, PR, CI, Error Analysis)
        csv_bytes = (
            b"model,y_true,y_pred,y_score,accuracy\n"
            b"Proposed,1,1,0.95,0.92\n"
            b"Proposed,1,1,0.88,0.91\n"
            b"Proposed,0,0,0.12,0.89\n"
            b"Proposed,0,1,0.65,0.85\n"
        )
        ds_info, df = register_uploaded_dataset(proj, csv_bytes, "exp_scores.csv")
        proj.experiment_results.datasets.append(ds_info)
        sync_default_experiment_records(proj)
        roc_pr = compute_roc_and_pr_curves(df, "y_true", "y_score", positive_label="1")
        assert 0.5 <= roc_pr["auroc"] <= 1.0
        assert 0.0 <= roc_pr["auprc"] <= 1.0
        stat_df = compute_statistical_summary(df, "model", "accuracy")
        assert not stat_df.empty and "95% CI (+/-)" in stat_df.columns
        err_df = extract_error_analysis_rows(df, "y_true", "y_pred")
        assert len(err_df) == 1
        assert "Measured Dataset" in explain_actual_results(proj)

        # 8. Manuscript Intelligence (non-destructive on approved sections)
        proj.manuscript_sections["Abstract"].content = "Approved abstract stays locked. Transformer outperforms baselines [1]."
        proj.manuscript_sections["Abstract"].approved = True
        wa = analyze_writing_quality("Abstract", proj.manuscript_sections["Abstract"].content)
        assert wa["word_count"] > 0
        mapped_claims = map_claims_to_evidence(proj, vs)
        assert len(mapped_claims) >= 1
        comp_rows = analyze_manuscript_completeness(proj)
        assert len(comp_rows) == 10
        repro = check_research_reproducibility(proj)
        assert 0.0 <= repro.score <= 100.0
        ac = run_academic_consistency_intelligence(proj)
        assert "rq_to_objective_map" in ac

        # 9. Publication Suitability (mocked OpenAlex)
        with patch(
            "research_os.publication_intel.discover_target_journals",
            return_value=([{"name": "IEEE TKDE", "type": "journal", "issn": "1041-4347", "is_oa": "Hybrid", "works_count": 5000, "cited_by_count": 200000, "homepage": "https://ieee.org"}], None),
        ):
            pub_items, err = build_publication_suitability_report(proj)
        assert not err and len(pub_items) == 1 and pub_items[0].verification_required is True

        # 10. LangGraph Contextual Workflow Assistant (must not mutate approved Abstract)
        ans = ask_project_workflow_assistant(proj, "Which research gaps have the strongest evidence?", vector_store=vs)
        assert "Strongest Evidence Gaps" in ans["answer"]
        assert proj.manuscript_sections["Abstract"].content == "Approved abstract stays locked. Transformer outperforms baselines [1]."

        # 11. Round-trip 2.0 state in .riq
        riq_bytes = export_project_to_riq_bytes(proj)
        restored, _, _ = validate_and_import_riq_bytes(riq_bytes)
        try:
            assert len(restored.gap_validation_reports) == 1
            assert len(restored.problem_candidates) == 3
            assert len(restored.methodology_recommendations) >= 2
            assert restored.reproducibility_report is not None
            assert len(restored.publication_suitability) == 1
        finally:
            cleanup_project_temp_dir(restored.project_id)
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_legacy_v1_riq_archive_backward_compatibility() -> None:
    """Verify older v1 .riq manifests without 2.0 fields import cleanly."""
    legacy_manifest = {
        "riq_format_version": "1.0.0",
        "project": {
            "project_id": "legacy_proj_01",
            "title": "Legacy v1 Study",
            "domain": "Computer Vision",
            "active_stage": "proposal",
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(legacy_manifest))
    restored, _, _ = validate_and_import_riq_bytes(buf.getvalue())
    try:
        assert restored.title == "Legacy v1 Study"
        assert restored.evidence_registry == []
        assert restored.problem_candidates == []
        assert restored.reproducibility_report is None
    finally:
        cleanup_project_temp_dir(restored.project_id)


def test_streamlit_ui_research_os_2_0_all_modules() -> None:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert not at.exception

    # Confirm all existing workspaces still exist plus Research OS 2.0
    nav_radio = at.radio[0]
    assert "Command Dashboard" in nav_radio.options
    assert "Workspace A — Research Intelligence" in nav_radio.options
    assert "Workspace B — Paper Studio" in nav_radio.options
    assert "Research OS 2.0 — Intelligence Hub" in nav_radio.options

    nav_radio.set_value("Research OS 2.0 — Intelligence Hub").run()
    assert not at.exception

    os_sel = next(sb for sb in at.selectbox if sb.key == "os2_module_select")
    for opt in list(os_sel.options):
        os_sel = next(sb for sb in at.selectbox if sb.key == "os2_module_select")
        os_sel.set_value(opt).run()
        assert not at.exception, f"Research OS 2.0 module '{opt}' failed: {at.exception}"
