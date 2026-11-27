"""Comprehensive end-to-end test suite covering every module, visualization, studio stage, and Streamlit UI view."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agents.paper_writer import build_reference_list, draft_manuscript_section, improve_or_rewrite_section
from core.config import MANUSCRIPT_SECTIONS, WORKFLOW_STAGES
from core.llm_client import GeminiClient
from core.project_manager import cleanup_project_temp_dir, create_empty_project
from core.schemas import ResearchGap, ScholarlyWork
from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from document_engine.figure_extractor import extract_figures_and_equations
from document_engine.table_extractor import extract_tables_from_pdf
from frontend.visualization_components import (
    fig_confusion_matrix,
    fig_experiment_comparison,
    fig_gap_classification_counts,
    fig_gap_feasibility_impact,
    fig_opportunity_radar,
    fig_paper_results_chart,
    fig_paper_structure_tree,
    fig_prisma_flow,
    fig_publication_timeline,
    fig_trends_heatmap,
)
from paper_studio.experiment_designer import generate_experiment_design
from paper_studio.implementation_helper import generate_implementation_plan
from paper_studio.literature_review import build_existing_vs_proposed_matrix
from paper_studio.manuscript_manager import compute_manuscript_completion, compute_manuscript_health
from paper_studio.methodology_designer import generate_methodology_design
from paper_studio.proposal_builder import generate_research_proposal
from paper_studio.publisher_exporter import assess_paper_readiness, export_latex
from paper_studio.topic_generator import generate_problem_statement, generate_research_topics
from research.citation_network import build_citation_graph, render_pyvis_html
from research.citation_verifier import extract_claims_from_manuscript
from research.multi_paper import compare_papers
from research.paper_intelligence import analyze_paper_intelligence
from research.research_gap import build_evidence_to_gap_graph, detect_research_gaps
from research.systematic_review import (
    apply_rule_based_screening,
    compute_prisma_counts,
    populate_literature_matrix_row,
    synthesize_literature_evidence,
)
from research.trend_analyzer import compute_research_trends
from tests.test_document_engine import _make_sample_pdf_bytes


def test_all_research_features_and_visualizations() -> None:
    proj = create_empty_project(title="End-to-End Research Test", domain="Artificial Intelligence")
    try:
        pdf_bytes = _make_sample_pdf_bytes()
        p1 = process_pdf_bytes(pdf_bytes, "paper_a.pdf")
        p2 = process_pdf_bytes(pdf_bytes, "paper_b.pdf")
        p2.title = "Second Comparative Study on Retrieval"
        proj.parsed_papers = [p1, p2]

        # Extractors
        _ = extract_tables_from_pdf(pdf_bytes)
        _ = extract_figures_and_equations(p1.pages)

        # Feature 1: Paper Intelligence
        a1 = analyze_paper_intelligence(p1, llm_client=None)
        a2 = analyze_paper_intelligence(p2, llm_client=None)
        proj.paper_analyses = {p1.paper_id: a1, p2.paper_id: a2}
        assert fig_paper_structure_tree(p1).data
        assert fig_paper_results_chart(a1) is not None

        # Feature 2: Research Gap Detective
        gaps = detect_research_gaps(proj.parsed_papers, proj.paper_analyses)
        assert len(gaps) >= 2
        proj.research_gaps = gaps
        assert fig_gap_feasibility_impact(gaps).data
        assert fig_gap_classification_counts(gaps).data
        gap_graph = build_evidence_to_gap_graph(gaps)
        assert gap_graph.number_of_nodes() >= 2

        # Feature 3: Multi-Paper Intelligence
        comp = compare_papers(proj.parsed_papers, proj.paper_analyses, proj.research_gaps)
        assert len(comp.methodology_matrix) == 2

        # Feature 5 & 8: Systematic Review + Trend Analyzer
        proj.discovered_works = [
            ScholarlyWork(
                work_id="w1",
                source="openalex",
                title="Transformer RAG for Science",
                authors=["A. Author"],
                year=2023,
                fields_of_study=["Natural Language Processing", "Information Retrieval"],
                abstract="We propose a Transformer RAG model evaluated on BioASQ achieving 89% accuracy.",
                citation_count=25,
                screening_status="unscreened",
            ),
            ScholarlyWork(
                work_id="w2",
                source="arxiv",
                title="Older Baseline Study",
                authors=["B. Author"],
                year=2015,
                fields_of_study=["Machine Learning"],
                abstract="Classical baseline study.",
                screening_status="unscreened",
            ),
        ]
        apply_rule_based_screening(proj.discovered_works, min_year=2018, required_keywords=["transformer"])
        assert proj.discovered_works[0].screening_status == "included"
        assert proj.discovered_works[1].screening_status == "excluded"

        populate_literature_matrix_row(proj.discovered_works[0])
        assert proj.discovered_works[0].extracted_methodology != ""

        prisma = compute_prisma_counts(proj.discovered_works, proj.parsed_papers)
        assert fig_prisma_flow(prisma).data
        synth = synthesize_literature_evidence(proj.discovered_works)
        assert synth["total_synthesized"] >= 1

        trends = compute_research_trends(proj.discovered_works)
        assert trends["total_works"] == 2
        assert fig_publication_timeline(proj.discovered_works).data
        assert fig_trends_heatmap(trends["topic_year_matrix"]).data
        assert fig_opportunity_radar(trends["opportunity_radar"]).data

        # Feature 6: PyVis Citation Network
        G = build_citation_graph(proj.discovered_works, proj.parsed_papers, proj.paper_analyses)
        html_net = render_pyvis_html(G, search_highlight="Transformer")
        assert "<html" in html_net.lower()
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_all_paper_studio_stages_1_to_12() -> None:
    proj = create_empty_project(title="Studio Pipeline Study", domain="Healthcare")
    try:
        llm = GeminiClient(api_key="")
        vs = SessionVectorStore(proj.project_id)

        # Stage 1 & 2
        topics = generate_research_topics("Healthcare", ["Artificial Intelligence"], "Clinical QA", proj, llm_client=llm)
        assert len(topics) >= 2
        proj.topic_ideas = topics
        proj.selected_topic = topics[0].title

        ps = generate_problem_statement(proj.selected_topic, proj, llm_client=llm)
        ps.approved = True
        proj.problem_statement = ps
        assert len(ps.objectives) >= 2

        # Stage 3 & 4 matrix
        proj.research_gaps.append(
            ResearchGap(
                title="Low clinical calibration",
                description="Needs calibration.",
                supporting_publication="Demo Paper",
                page_reference="p. 2",
                potential_improvement="Add conformal calibration",
            )
        )
        mat = build_existing_vs_proposed_matrix(proj)
        assert len(mat) == 1

        # Stage 5, 6, 7, 8
        prop = generate_research_proposal(proj, llm_client=llm)
        prop.approved = True
        proj.proposal = prop
        assert len(prop.milestones) >= 3

        meth = generate_methodology_design(proj, llm_client=llm)
        meth.approved = True
        proj.methodology = meth
        assert "flowchart" in meth.mermaid_architecture

        imp = generate_implementation_plan(proj, llm_client=llm)
        assert "STARTING POINT" in imp.starter_code
        proj.implementation = imp

        exp_d = generate_experiment_design(proj, llm_client=llm)
        assert len(exp_d.baselines) >= 2
        proj.experiment_design = exp_d

        # Stage 9 charts
        df = pd.DataFrame({"Model": ["Base", "Ours"], "Accuracy": [0.78, 0.89]})
        assert fig_experiment_comparison(df, "Model", ["Accuracy"]).data
        assert fig_confusion_matrix([[5, 1], [0, 6]], ["0", "1"]).data

        # Stage 10 & 11 manuscript drafting, improvement, claims, health
        for sec in MANUSCRIPT_SECTIONS:
            draft = draft_manuscript_section(proj, sec, [], llm_client=llm)
            proj.manuscript_sections[sec].content = draft
            proj.manuscript_sections[sec].approved = True

        improved = improve_or_rewrite_section(
            "Abstract",
            proj.manuscript_sections["Abstract"].content,
            "add_evidence",
            [{"paper_title": "Demo Paper", "pages": "pp. 1-2", "text": "Evidence snippet."}],
            llm_client=llm,
        )
        assert "Supporting Session Evidence" in improved

        proj.manuscript_sections["Introduction"].content += " Prior work confirms this gain [1]."
        claims = extract_claims_from_manuscript(proj.manuscript_sections)
        assert len(claims) >= 1

        comp_ms = compute_manuscript_completion(proj)
        assert comp_ms["approved_count"] == len(MANUSCRIPT_SECTIONS)
        health = compute_manuscript_health(proj)
        assert health["consistency"].overall_score >= 0.0

        # Stage 12 readiness & Springer/ACM LaTeX
        proj.publication_prep.target_format = "Springer"
        readiness = assess_paper_readiness(proj)
        assert readiness["readiness_score"] > 50.0
        assert "\\documentclass[runningheads]{llncs}" in export_latex(proj)

        proj.publication_prep.target_format = "ACM"
        assert "\\documentclass[sigconf]{acmart}" in export_latex(proj)
        assert build_reference_list(proj, style="ACM") is not None
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_streamlit_ui_all_workspaces_and_modules() -> None:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    assert not at.exception, f"Dashboard exception: {at.exception}"

    # Switch to Workspace A and iterate through every feature module (1..10)
    at.radio[0].set_value("Workspace A — Research Intelligence").run()
    assert not at.exception, f"Workspace A initial exception: {at.exception}"

    ws_a_select = next(sb for sb in at.selectbox if sb.key == "ws_a_module_select")
    for option in list(ws_a_select.options):
        ws_a_select = next(sb for sb in at.selectbox if sb.key == "ws_a_module_select")
        ws_a_select.set_value(option).run()
        assert not at.exception, f"Workspace A module '{option}' failed: {at.exception}"

    # Switch to Workspace B and visit all 12 Paper Studio stages
    at.radio[0].set_value("Workspace B — Paper Studio").run()
    assert not at.exception, f"Workspace B initial exception: {at.exception}"

    for stg in WORKFLOW_STAGES:
        btn = next(b for b in at.button if b.key == f"st_nav_{stg['id']}")
        btn.click().run()
        assert not at.exception, f"Paper Studio stage '{stg['id']}' failed: {at.exception}"


def test_title_improvement_and_live_docx_sync() -> None:
    import io
    import docx
    from paper_studio.live_docx_builder import rebuild_live_docx, sync_stage_to_word_doc
    from paper_studio.topic_generator import suggest_improved_titles

    proj = create_empty_project(title="Untitled Research Project", domain="Cybersecurity")
    try:
        critique, suggestions = suggest_improved_titles("Phishing Detection", "Cybersecurity")
        assert critique
        assert len(suggestions) == 3

        # User retains their own title
        proj.original_title_input = "Phishing Detection"
        proj.title = "Phishing Detection"
        proj.selected_topic = "Phishing Detection"
        proj.proposal.title = "Phishing Detection"
        proj.title_confirmed = True

        updated_t = sync_stage_to_word_doc(proj, "title")
        assert "Abstract" in updated_t

        # Add Problem & Methodology components -> verify live .docx grows and reflects them
        proj.problem_statement = generate_problem_statement(proj.selected_topic, proj)
        updated_p = sync_stage_to_word_doc(proj, "problem")
        assert "Introduction" in updated_p
        assert "Objectives" in updated_p

        # Lock Introduction so subsequent syncs respect user's approved prose
        proj.manuscript_sections["Introduction"].content = "Custom locked introduction."
        proj.manuscript_sections["Introduction"].approved = True
        sync_stage_to_word_doc(proj, "problem", force=False)
        assert proj.manuscript_sections["Introduction"].content == "Custom locked introduction."

        docx_bytes = rebuild_live_docx(proj)
        doc = docx.Document(io.BytesIO(docx_bytes))
        full_doc_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Phishing Detection" in full_doc_text
        assert "Custom locked introduction." in full_doc_text
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_workspace_b_title_retain_and_inline_word_edit_ui() -> None:
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=25).run()
    at.radio[0].set_value("Workspace B — Paper Studio").run()
    assert not at.exception

    # Enter user's own title and evaluate improvements
    title_in = next(ti for ti in at.text_input if ti.key == "s1_user_title_input")
    title_in.set_value("Edge Intrusion Detection").run()
    eval_btn = next(b for b in at.button if b.key == "s1_eval_title_btn")
    eval_btn.click().run()
    assert not at.exception

    # Retain own title (first radio option) and confirm
    choice_radio = next(r for r in at.radio if r.key == "s1_title_choice_radio")
    assert choice_radio.options[0].startswith("Keep my own title:")
    choice_radio.set_value(choice_radio.options[0]).run()
    confirm_btn = next(b for b in at.button if b.key == "s1_confirm_title_btn")
    confirm_btn.click().run()
    assert not at.exception

    # Alter Abstract directly inside Right Panel Live Word Doc editor
    inline_ed = next(ta for ta in at.text_area if ta.key == "rp_docx_inline_Abstract")
    inline_ed.set_value("Directly altered live Word abstract.").run()
    assert not at.exception
    assert at.session_state["projects"][at.session_state["active_project_id"]].manuscript_sections["Abstract"].content == "Directly altered live Word abstract."

