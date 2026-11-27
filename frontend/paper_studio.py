"""Workspace B — Three-Panel Research Paper Studio across Stages 1-12."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from agents.paper_writer import build_reference_list, draft_manuscript_section, improve_or_rewrite_section
from core.config import MANUSCRIPT_SECTIONS, WORKFLOW_STAGES
from core.llm_client import GeminiClient, get_llm_client
from core.project_manager import export_project_to_riq_bytes, sanitize_filename
from core.schemas import ManuscriptSectionState, ResearchProject, utc_now_iso
from core.vector_store import SessionVectorStore
from frontend.components import (
    render_gap_card,
    render_metric_card,
    render_project_status_bar,
    render_section_header,
)
from frontend.visualization_components import (
    PLOTLY_LAYOUT_BASE,
    fig_confusion_matrix,
    fig_experiment_comparison,
    fig_gap_feasibility_impact,
    fig_prisma_flow,
    render_mermaid_diagram,
)
from paper_studio.experiment_designer import generate_experiment_design
from paper_studio.implementation_helper import generate_implementation_plan
from paper_studio.literature_review import build_existing_vs_proposed_matrix
from paper_studio.manuscript_manager import compute_manuscript_completion, compute_manuscript_health
from paper_studio.methodology_designer import generate_methodology_design
from paper_studio.proposal_builder import generate_research_proposal
from paper_studio.publisher_exporter import (
    assess_paper_readiness,
    discover_target_journals,
    export_bibtex,
    export_docx_bytes,
    export_latex,
    export_pdf_bytes,
)
from paper_studio.results_analyzer import (
    compute_classification_evaluation,
    load_project_dataset_df,
    register_uploaded_dataset,
)
from paper_studio.topic_generator import generate_problem_statement, generate_research_topics
from research.citation_verifier import extract_claims_from_manuscript, verify_claim_against_evidence
from research.consistency_checker import run_consistency_check
from research.cross_domain import COMMON_DOMAINS
from research.systematic_review import compute_prisma_counts


def render_paper_studio(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    assets_cache: dict[str, bytes],
) -> None:
    """Render Workspace B in a Three-Panel guided layout (Left Journey | Center Studio | Right Evidence)."""
    render_project_status_bar(project, last_autosave_ts=st.session_state.get("last_autosave_ts", ""))
    llm_client = get_llm_client(st.session_state.get("gemini_api_key"))

    # Top Studio Save Action
    top_l, top_r = st.columns([3.2, 1.0])
    with top_l:
        render_section_header(
            "Research Paper Studio",
            "Three-panel visual development environment: edit and approve each stage before feeding it into downstream stages.",
            badge="WORKSPACE B",
        )
    with top_r:
        riq_data = export_project_to_riq_bytes(project, asset_bytes_map=assets_cache)
        st.download_button(
            "Save Project (.riq)",
            data=riq_data,
            file_name=f"{sanitize_filename(project.title)}.riq",
            mime="application/zip",
            use_container_width=True,
            key="studio_save_riq_top",
        )

    left_col, center_col, right_col = st.columns([1.05, 2.55, 1.2], gap="medium")

    # -----------------------------------------------------------------------
    # LEFT PANEL: Research Journey, Section Navigation, Completion & Approval
    # -----------------------------------------------------------------------
    with left_col:
        with st.container(border=True):
            st.markdown("#### Research Journey")
            current_stage_num = int(st.session_state.get("studio_stage_num", 1))
            for stg in WORKFLOW_STAGES:
                snum = stg["studio_stage"]
                done = stg["id"] in project.completed_stages
                active = snum == current_stage_num
                badge = "✓" if done else ("●" if active else f"{snum}")
                label = f"{badge}  {stg['title']}"
                if st.button(
                    label,
                    key=f"st_nav_{stg['id']}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["studio_stage_num"] = snum
                    project.active_stage = stg["id"]
                    project.updated_at = utc_now_iso()
                    st.rerun()

            st.divider()
            comp = compute_manuscript_completion(project)
            st.markdown("#### Manuscript Status")
            st.progress(min(1.0, comp["completion_pct"] / 100.0))
            st.caption(f"Drafted: {comp['drafted_count']}/12 | Approved: {comp['approved_count']}/12")
            for sec_name in MANUSCRIPT_SECTIONS:
                sec = project.manuscript_sections.get(sec_name, ManuscriptSectionState(name=sec_name))
                icon = "✅" if sec.approved else ("📝" if sec.content.strip() else "○")
                if st.button(f"{icon} {sec_name}", key=f"st_sec_jump_{sec_name}", use_container_width=True):
                    st.session_state["studio_stage_num"] = 10
                    st.session_state["studio_active_ms_section"] = sec_name
                    project.active_stage = "manuscript"
                    st.rerun()

    # -----------------------------------------------------------------------
    # CENTER PANEL: Interactive Stage Workspace & Section Editor
    # -----------------------------------------------------------------------
    stage_num = int(st.session_state.get("studio_stage_num", 1))
    with center_col:
        with st.container(border=True):
            if stage_num == 1:
                _stage_1_idea(project, llm_client)
            elif stage_num == 2:
                _stage_2_problem(project, llm_client)
            elif stage_num == 3:
                _stage_3_literature(project)
            elif stage_num == 4:
                _stage_4_gaps(project)
            elif stage_num == 5:
                _stage_5_proposal(project, llm_client)
            elif stage_num == 6:
                _stage_6_methodology(project, llm_client)
            elif stage_num == 7:
                _stage_7_implementation(project, llm_client)
            elif stage_num == 8:
                _stage_8_experiments(project, llm_client)
            elif stage_num == 9:
                _stage_9_results(project, assets_cache)
            elif stage_num == 10:
                _stage_10_manuscript(project, vector_store, llm_client)
            elif stage_num == 11:
                _stage_11_validation(project, vector_store, llm_client)
            else:
                _stage_12_publication(project)

    # -----------------------------------------------------------------------
    # RIGHT PANEL: Evidence, Citations, Writing Tips & Consistency Warnings
    # -----------------------------------------------------------------------
    with right_col:
        with st.container(border=True):
            st.markdown("#### Research Evidence")
            rag_q = st.text_input(
                "Search indexed PDFs",
                value=project.selected_topic or project.domain,
                key="rp_rag_query",
            )
            if vector_store.chunks:
                hits = vector_store.search(rag_q, top_k=3, llm_client=llm_client)
                for chunk, score in hits:
                    with st.expander(f"{chunk.paper_title[:28]} (pp.{chunk.page_start}-{chunk.page_end})", expanded=False):
                        st.caption(f"Section: {chunk.section} | Score: {score:.2f}")
                        st.write(chunk.text[:320] + "...")
            else:
                st.caption("Upload PDFs in Workspace A to query page-level RAG evidence here.")

            st.divider()
            st.markdown("#### Citation Suggestions")
            refs = build_reference_list(project, style=project.publication_prep.target_format)
            if refs:
                for r in refs[:5]:
                    st.caption(r[:140])
            else:
                st.caption("Discover papers or upload PDFs to populate verified citation suggestions.")

            st.divider()
            st.markdown("#### Consistency Warnings")
            consistency = run_consistency_check(project)
            open_issues = [i for i in consistency.issues if not i.resolved]
            if not open_issues:
                st.success("No consistency warnings.")
            else:
                st.caption(f"Health Score: **{consistency.overall_score}/100**")
                for iss in open_issues[:4]:
                    st.warning(f"**{iss.category}**: {iss.description[:110]}")


def _mark_stage_done(project: ResearchProject, stage_id: str) -> None:
    if stage_id not in project.completed_stages:
        project.completed_stages.append(stage_id)
    project.unsaved_changes = True
    project.updated_at = utc_now_iso()


# ---------------------------------------------------------------------------
# STAGE IMPLEMENTATIONS (1 - 12)
# ---------------------------------------------------------------------------

def _stage_1_idea(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 1 — Research Idea & Topic Exploration")
    c1, c2 = st.columns(2)
    with c1:
        domain = st.selectbox(
            "Primary Domain",
            COMMON_DOMAINS,
            index=COMMON_DOMAINS.index(project.domain) if project.domain in COMMON_DOMAINS else 0,
            key="s1_domain",
        )
        project.domain = domain
    with c2:
        sec_domains = st.multiselect(
            "Secondary Domains (Cross-Domain)",
            [d for d in COMMON_DOMAINS if d != domain],
            default=[d for d in project.secondary_domains if d != domain],
            key="s1_sec_domains",
        )
        project.secondary_domains = sec_domains

    seed = st.text_input("Research Seed / Focus Area", value=project.selected_topic, placeholder="e.g., Reliable clinical retrieval-augmented QA", key="s1_seed")
    if st.button("Generate Research Topics & Questions", type="primary", key="s1_gen_btn"):
        project.topic_ideas = generate_research_topics(domain, sec_domains, seed, project, llm_client=llm_client)
        project.unsaved_changes = True

    if project.topic_ideas:
        # Topic cluster / opportunity scatter
        tdf = pd.DataFrame(
            [
                {
                    "Topic": t.title[:36],
                    "Feasibility": t.feasibility_score,
                    "Impact": t.impact_score,
                    "Novelty": t.novelty_level,
                }
                for t in project.topic_ideas
            ]
        )
        fig = px.scatter(tdf, x="Feasibility", y="Impact", color="Novelty", hover_name="Topic", title="Topic Opportunity Map")
        fig.update_traces(marker=dict(size=14))
        fig.update_layout(**PLOTLY_LAYOUT_BASE, height=280)
        st.plotly_chart(fig, use_container_width=True)

        for idx, t in enumerate(project.topic_ideas):
            with st.container(border=True):
                st.markdown(f"**{t.title}** (`{t.novelty_level}` | Feasibility {t.feasibility_score} | Impact {t.impact_score})")
                st.caption(t.rationale)
                for rq in t.research_questions:
                    st.markdown(f"- {rq}")
                if st.button("Select This Topic & Approve Stage 1", key=f"s1_pick_{idx}"):
                    project.selected_topic = t.title
                    project.title = t.title[:80]
                    _mark_stage_done(project, "idea")
                    st.success("Topic selected and Stage 1 marked complete.")
                    st.rerun()

    manual_topic = st.text_input("Or Enter Custom Approved Topic", value=project.selected_topic, key="s1_manual")
    if st.button("Approve Custom Topic", key="s1_approve_manual") and manual_topic.strip():
        project.selected_topic = manual_topic.strip()
        project.title = manual_topic.strip()[:80]
        _mark_stage_done(project, "idea")
        st.success("Stage 1 approved.")


def _stage_2_problem(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 2 — Research Problem Statement")
    topic = project.selected_topic or project.title
    if st.button("Generate Problem Statement & Objectives", type="primary", key="s2_gen_btn"):
        project.problem_statement = generate_problem_statement(topic, project, llm_client=llm_client)
        project.unsaved_changes = True

    ps = project.problem_statement
    ps.problem_definition = st.text_area("Problem Definition", value=ps.problem_definition, height=110, key="s2_prob_def")
    ps.motivation = st.text_area("Research Motivation", value=ps.motivation, height=85, key="s2_motiv")

    obj_text = st.text_area("Objectives (one per line)", value="\n".join(ps.objectives), height=110, key="s2_objs")
    ps.objectives = [ln.strip() for ln in obj_text.splitlines() if ln.strip()]

    c1, c2 = st.columns(2)
    with c1:
        rq_text = st.text_area("Research Questions (one per line)", value="\n".join(ps.research_questions), height=100, key="s2_rqs")
        ps.research_questions = [ln.strip() for ln in rq_text.splitlines() if ln.strip()]
    with c2:
        hyp_text = st.text_area("Hypotheses (one per line)", value="\n".join(ps.hypotheses), height=100, key="s2_hyps")
        ps.hypotheses = [ln.strip() for ln in hyp_text.splitlines() if ln.strip()]

    # Problem decomposition Mermaid tree
    if ps.objectives:
        lines = ["flowchart TD", f'    P["Problem: {topic[:34]}"]']
        for i, o in enumerate(ps.objectives[:5]):
            lines.append(f'    O{i}["{o[:38]}"]')
            lines.append(f"    P --> O{i}")
        render_mermaid_diagram("\n".join(lines), height=250, title="Problem & Objective Hierarchy")

    if st.button("Approve Problem Statement (Stage 2)", type="primary", key="s2_approve"):
        ps.approved = True
        _mark_stage_done(project, "discovery")
        st.success("Stage 2 approved and available to downstream stages.")


def _stage_3_literature(project: ResearchProject) -> None:
    st.markdown("### Stage 3 — Literature Review & Evidence Synthesis")
    prisma = compute_prisma_counts(project.discovered_works, project.parsed_papers)
    st.plotly_chart(fig_prisma_flow(prisma), use_container_width=True)

    included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
    st.caption(f"Included studies: {len(included)} | Uploaded PDFs: {len(project.parsed_papers)}")
    if included:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Title": w.title[:55],
                        "Year": w.year or "N/A",
                        "Methodology": w.extracted_methodology or "-",
                        "Limitations": w.extracted_limitations or "-",
                    }
                    for w in included
                ]
            ),
            use_container_width=True,
        )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Open Systematic Review Workbench", key="s3_open_ws_a"):
            st.session_state["active_nav"] = "Workspace A — Research Intelligence"
            st.session_state["workspace_a_tab"] = 4
            st.rerun()
    with c2:
        if st.button("Approve Literature Review (Stage 3)", type="primary", key="s3_approve"):
            _mark_stage_done(project, "literature_review")
            st.success("Stage 3 approved.")


def _stage_4_gaps(project: ResearchProject) -> None:
    st.markdown("### Stage 4 — Research Gap & Proposed Contribution")
    if project.research_gaps:
        st.plotly_chart(fig_gap_feasibility_impact(project.research_gaps), use_container_width=True)
        matrix = build_existing_vs_proposed_matrix(project)
        st.markdown("#### Existing Work vs Proposed Contribution Matrix")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True)
        for g in project.research_gaps[:4]:
            render_gap_card(g)
    else:
        st.info("No gaps detected yet. Run Gap Detective in Workspace A or upload PDFs.")
        if st.button("Go to Research Gap Detective", key="s4_go_gaps"):
            st.session_state["active_nav"] = "Workspace A — Research Intelligence"
            st.session_state["workspace_a_tab"] = 1
            st.rerun()

    if st.button("Approve Gap & Contribution Framing (Stage 4)", type="primary", key="s4_approve"):
        _mark_stage_done(project, "gap_identification")
        st.success("Stage 4 approved.")


def _stage_5_proposal(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 5 — Research Proposal & Roadmap")
    if st.button("Generate Research Proposal", type="primary", key="s5_gen_btn"):
        project.proposal = generate_research_proposal(project, llm_client=llm_client)
        project.unsaved_changes = True

    prop = project.proposal
    prop.title = st.text_input("Proposal Title", value=prop.title or project.selected_topic or project.title, key="s5_title")
    prop.background = st.text_area("Background", value=prop.background, height=90, key="s5_bg")
    prop.methodology_summary = st.text_area("Methodology Summary", value=prop.methodology_summary, height=95, key="s5_meth")
    prop.scope = st.text_input("Scope", value=prop.scope, key="s5_scope")

    if prop.milestones:
        st.markdown("#### Research Roadmap & Deliverable Tracker")
        st.dataframe(pd.DataFrame(prop.milestones), use_container_width=True)

    if st.button("Approve Research Proposal (Stage 5)", type="primary", key="s5_approve"):
        prop.approved = True
        _mark_stage_done(project, "proposal")
        st.success("Stage 5 approved.")


def _stage_6_methodology(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 6 — Methodology & System Architecture")
    if st.button("Generate Architecture & Diagrams", type="primary", key="s6_gen_btn"):
        project.methodology = generate_methodology_design(project, llm_client=llm_client)
        project.saved_diagrams["architecture"] = project.methodology.mermaid_architecture
        project.saved_diagrams["workflow"] = project.methodology.mermaid_workflow
        project.saved_diagrams["data_pipeline"] = project.methodology.mermaid_data_pipeline
        project.unsaved_changes = True

    md = project.methodology
    md.proposed_architecture = st.text_area("Proposed Architecture Narrative", value=md.proposed_architecture, height=100, key="s6_arch")
    md.mathematical_models = st.text_area("Mathematical Formulation", value=md.mathematical_models, height=75, key="s6_math")

    d_tab1, d_tab2, d_tab3 = st.tabs(["System Architecture", "Workflow Diagram", "Data Pipeline"])
    with d_tab1:
        md.mermaid_architecture = st.text_area("Edit Architecture Mermaid", value=md.mermaid_architecture, height=115, key="s6_m_arch")
        render_mermaid_diagram(md.mermaid_architecture, height=300)
        st.download_button("Download Architecture (.mmd)", data=md.mermaid_architecture or "flowchart LR\nA-->B", file_name="architecture.mmd", key="s6_dl_arch")
    with d_tab2:
        md.mermaid_workflow = st.text_area("Edit Workflow Mermaid", value=md.mermaid_workflow, height=115, key="s6_m_wf")
        render_mermaid_diagram(md.mermaid_workflow, height=300)
        st.download_button("Download Workflow (.mmd)", data=md.mermaid_workflow or "flowchart TD\nA-->B", file_name="workflow.mmd", key="s6_dl_wf")
    with d_tab3:
        md.mermaid_data_pipeline = st.text_area("Edit Pipeline Mermaid", value=md.mermaid_data_pipeline, height=115, key="s6_m_pipe")
        render_mermaid_diagram(md.mermaid_data_pipeline, height=300)
        st.download_button("Download Pipeline (.mmd)", data=md.mermaid_data_pipeline or "flowchart LR\nA-->B", file_name="pipeline.mmd", key="s6_dl_pipe")

    if st.button("Approve Methodology & Architecture (Stage 6)", type="primary", key="s6_approve"):
        project.saved_diagrams["architecture"] = md.mermaid_architecture
        project.saved_diagrams["workflow"] = md.mermaid_workflow
        project.saved_diagrams["data_pipeline"] = md.mermaid_data_pipeline
        md.approved = True
        _mark_stage_done(project, "methodology")
        st.success("Stage 6 approved.")


def _stage_7_implementation(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 7 — Implementation Blueprint & Starter Code")
    st.warning("Generated starter code is a starting point only and requires thorough testing and validation on real data.")
    if st.button("Generate Implementation Plan & Starter Code", type="primary", key="s7_gen_btn"):
        project.implementation = generate_implementation_plan(project, llm_client=llm_client)
        project.unsaved_changes = True

    imp = project.implementation
    if imp.tech_stack:
        st.markdown("#### Technology Selection")
        st.dataframe(pd.DataFrame(imp.tech_stack), use_container_width=True)
    if imp.modules:
        st.markdown("#### Module Dependency Plan")
        st.dataframe(pd.DataFrame(imp.modules), use_container_width=True)

    imp.starter_code = st.text_area("Starter Code (Starting Point — Validate Before Use)", value=imp.starter_code, height=220, key="s7_code")
    imp.documentation_notes = st.text_area("Implementation Documentation Notes", value=imp.documentation_notes, height=80, key="s7_notes")

    if imp.checklist:
        st.markdown("#### Implementation Checklist")
        for idx, item in enumerate(imp.checklist):
            checked = st.checkbox(item.get("task", f"Task {idx+1}"), value=bool(item.get("done")), key=f"s7_chk_{idx}")
            item["done"] = checked

    if st.button("Approve Implementation Stage (Stage 7)", type="primary", key="s7_approve"):
        imp.approved = True
        _mark_stage_done(project, "implementation")
        st.success("Stage 7 approved.")


def _stage_8_experiments(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Stage 8 — Experiment Design & Ablation Plan")
    if st.button("Generate Experiment & Ablation Plan", type="primary", key="s8_gen_btn"):
        project.experiment_design = generate_experiment_design(project, llm_client=llm_client)
        project.unsaved_changes = True

    ed = project.experiment_design
    if ed.baselines:
        st.markdown("#### Baselines")
        st.dataframe(pd.DataFrame(ed.baselines), use_container_width=True)
    if ed.ablation_plan:
        st.markdown("#### Ablation Study Matrix")
        st.dataframe(pd.DataFrame(ed.ablation_plan), use_container_width=True)
    if ed.resource_estimates:
        rdf = pd.DataFrame(ed.resource_estimates)
        if "stage" in rdf.columns and "estimated_hours" in rdf.columns:
            fig = px.bar(rdf, x="stage", y="estimated_hours", color="compute" if "compute" in rdf.columns else None, title="Estimated Experiment Effort (Hours)")
            fig.update_layout(**PLOTLY_LAYOUT_BASE, height=290)
            st.plotly_chart(fig, use_container_width=True)

    metrics_str = st.text_input("Evaluation Metrics (comma-separated)", value=", ".join(ed.evaluation_metrics), key="s8_metrics")
    ed.evaluation_metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]

    if st.button("Approve Experiment Design (Stage 8)", type="primary", key="s8_approve"):
        ed.approved = True
        _mark_stage_done(project, "experiments")
        st.success("Stage 8 approved.")


def _stage_9_results(project: ResearchProject, assets_cache: dict[str, bytes]) -> None:
    st.markdown("### Stage 9 — Results & Discussion (Actual Experimental Data Only)")
    st.caption("Upload real CSV or Excel results. RESEARCHIQ computes metrics directly from your supplied data and never fabricates measurements.")

    up_data = st.file_uploader("Upload Experimental Results (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"], key="s9_data_uploader")
    if up_data is not None and st.button("Ingest Uploaded Dataset", type="primary", key="s9_ingest_btn"):
        try:
            raw = up_data.getvalue()
            ds_info, _ = register_uploaded_dataset(project, raw, up_data.name)
            assets_cache[ds_info.asset_relpath] = raw
            project.experiment_results.datasets.append(ds_info)
            project.unsaved_changes = True
            st.success(f"Loaded '{ds_info.filename}' ({ds_info.row_count} rows, {len(ds_info.columns)} columns).")
        except Exception as exc:
            st.error(f"Dataset error: {exc}")

    er = project.experiment_results
    if not er.datasets:
        st.info("Upload a CSV or Excel file containing your model/experiment metrics or predictions.")
        return

    sel_ds = st.selectbox("Select Dataset", er.datasets, format_func=lambda d: f"{d.filename} ({d.row_count} rows)", key="s9_sel_ds")
    df = load_project_dataset_df(project, sel_ds, assets_cache)
    if df is None or df.empty:
        st.warning("Could not load dataset rows.")
        return

    st.dataframe(df.head(30), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    if numeric_cols and all_cols:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Model / Condition Column (X-axis)", all_cols, key="s9_x_col")
        with c2:
            y_cols = st.multiselect("Metric Columns (Y-axis)", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))], key="s9_y_cols")
        er.selected_model_col = x_col
        er.selected_metric_cols = y_cols
        st.plotly_chart(fig_experiment_comparison(df, x_col, y_cols), use_container_width=True)

    with st.expander("Compute Classification Metrics & Confusion Matrix (from True/Pred columns)", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            yt_col = st.selectbox("Ground Truth Column", all_cols, key="s9_yt")
        with cc2:
            yp_col = st.selectbox("Prediction Column", all_cols, index=min(1, len(all_cols) - 1), key="s9_yp")
        if st.button("Compute Real Classification Metrics", key="s9_calc_cm"):
            try:
                res_cm = compute_classification_evaluation(df, yt_col, yp_col)
                er.computed_classification_metrics = res_cm["metrics"]
                er.confusion_matrix = res_cm["confusion_matrix"]
                er.confusion_labels = res_cm["labels"]
                project.unsaved_changes = True
            except Exception as exc:
                st.error(str(exc))

        if er.computed_classification_metrics:
            mcols = st.columns(4)
            mets = er.computed_classification_metrics
            with mcols[0]:
                render_metric_card("Accuracy", mets.get("accuracy", "-"), f"N={mets.get('samples', 0)}")
            with mcols[1]:
                render_metric_card("Precision (Macro)", mets.get("precision_macro", "-"))
            with mcols[2]:
                render_metric_card("Recall (Macro)", mets.get("recall_macro", "-"))
            with mcols[3]:
                render_metric_card("F1-Score (Macro)", mets.get("f1_macro", "-"))
            st.plotly_chart(fig_confusion_matrix(er.confusion_matrix, er.confusion_labels), use_container_width=True)

    er.discussion_notes = st.text_area("Results Interpretation & Discussion Notes", value=er.discussion_notes, height=110, key="s9_disc")
    if st.button("Approve Verified Results (Stage 9)", type="primary", key="s9_approve"):
        er.approved = True
        _mark_stage_done(project, "results")
        st.success("Stage 9 approved.")


def _stage_10_manuscript(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Stage 10 — Independent Section-Wise Manuscript Generation")
    default_sec = st.session_state.get("studio_active_ms_section", MANUSCRIPT_SECTIONS[0])
    if default_sec not in MANUSCRIPT_SECTIONS:
        default_sec = MANUSCRIPT_SECTIONS[0]
    sec_name = st.selectbox("Select Manuscript Section", MANUSCRIPT_SECTIONS, index=MANUSCRIPT_SECTIONS.index(default_sec), key="s10_sec_select")
    st.session_state["studio_active_ms_section"] = sec_name

    sec_state = project.manuscript_sections.setdefault(sec_name, ManuscriptSectionState(name=sec_name))
    instruction = st.text_input("Optional Section Guidance", placeholder="e.g., Emphasize ablation gains and statistical variance", key="s10_instr")

    a1, a2, a3, a4, a5 = st.columns(5)
    hits = vector_store.search(f"{project.selected_topic or project.title} {sec_name}", top_k=4, llm_client=llm_client) if vector_store.chunks else []
    ev_dicts = [{"paper_title": c.paper_title, "pages": f"pp. {c.page_start}-{c.page_end}", "text": c.text} for c, _ in hits]

    with a1:
        if st.button("Generate Draft", type="primary", use_container_width=True, key="s10_gen"):
            sec_state.content = draft_manuscript_section(project, sec_name, ev_dicts, llm_client=llm_client, instruction_override=instruction)
            sec_state.last_updated = utc_now_iso()
            project.unsaved_changes = True
            st.rerun()
    with a2:
        if st.button("Improve Flow", use_container_width=True, key="s10_imp"):
            sec_state.content = improve_or_rewrite_section(sec_name, sec_state.content, "improve", ev_dicts, llm_client=llm_client, custom_note=instruction)
            project.unsaved_changes = True
            st.rerun()
    with a3:
        if st.button("Rewrite", use_container_width=True, key="s10_rew"):
            sec_state.content = improve_or_rewrite_section(sec_name, sec_state.content, "rewrite", ev_dicts, llm_client=llm_client, custom_note=instruction)
            project.unsaved_changes = True
            st.rerun()
    with a4:
        if st.button("Add Evidence", use_container_width=True, key="s10_evi"):
            sec_state.content = improve_or_rewrite_section(sec_name, sec_state.content, "add_evidence", ev_dicts, llm_client=llm_client)
            project.unsaved_changes = True
            st.rerun()
    with a5:
        if st.button("Approve ✓" if not sec_state.approved else "Approved ✅", use_container_width=True, key="s10_app"):
            sec_state.approved = not sec_state.approved
            if sec_state.approved and all(project.manuscript_sections[s].approved for s in MANUSCRIPT_SECTIONS):
                _mark_stage_done(project, "manuscript")
            project.unsaved_changes = True
            st.rerun()

    edited = st.text_area(f"{sec_name} Editor", value=sec_state.content, height=340, key=f"s10_editor_{sec_name}")
    if edited != sec_state.content:
        sec_state.content = edited
        sec_state.word_count = len(edited.split())
        sec_state.last_updated = utc_now_iso()
        project.unsaved_changes = True

    st.caption(f"Word count: **{len(sec_state.content.split())}** | Status: **{'Approved' if sec_state.approved else 'Draft (Unapproved)'}**")


def _stage_11_validation(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Stage 11 — Comprehensive Manuscript Validation")
    if st.button("Run Full Validation Suite (Citations + Consistency + References)", type="primary", key="s11_run"):
        extracted = extract_claims_from_manuscript(project.manuscript_sections)
        if extracted:
            project.claim_verifications = [
                verify_claim_against_evidence(clm, marker, sec, vector_store, project.discovered_works, llm_client)
                for sec, clm, marker in extracted
            ]
        project.consistency_report = run_consistency_check(project)
        project.unsaved_changes = True

    health = compute_manuscript_health(project)
    comp = health["completion"]
    consistency = health["consistency"]
    ref_chk = health["reference_check"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Completion", f"{comp['completion_pct']}%", f"{comp['approved_count']}/12 approved")
    with c2:
        render_metric_card("Consistency", f"{consistency.overall_score}/100", f"{len(consistency.issues)} issues")
    with c3:
        render_metric_card("References", ref_chk["reference_entries_count"], "Complete" if ref_chk["is_complete"] else "Needs attention")
    with c4:
        render_metric_card("Total Words", comp["total_words"], "Across all sections")

    sec_df = pd.DataFrame(comp["section_rows"])
    fig = px.bar(sec_df, x="Section", y="Words", color="Approved", title="Section Word Count & Approval Status", color_discrete_map={"Approved": "#10B981", "Pending": "#8B5CF6"})
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=320)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Approve Validation Stage (Stage 11)", type="primary", key="s11_approve"):
        _mark_stage_done(project, "review")
        st.success("Stage 11 approved.")


def _stage_12_publication(project: ResearchProject) -> None:
    st.markdown("### Stage 12 — Publication Formatting, Journal Discovery & Export")
    pp = project.publication_prep

    c1, c2 = st.columns(2)
    with c1:
        fmt = st.selectbox("Target Publication Style", ["IEEE", "Springer", "ACM"], index=["IEEE", "Springer", "ACM"].index(pp.target_format), key="s12_fmt")
        pp.target_format = fmt  # type: ignore[assignment]
        authors_in = st.text_input("Authors (comma-separated)", value=", ".join(pp.authors), key="s12_authors")
        pp.authors = [a.strip() for a in authors_in.split(",") if a.strip()]
    with c2:
        pp.affiliation = st.text_input("Affiliation", value=pp.affiliation, key="s12_affil")
        kw_in = st.text_input("Keywords (comma-separated)", value=", ".join(pp.keywords), key="s12_kw")
        pp.keywords = [k.strip() for k in kw_in.split(",") if k.strip()]

    readiness = assess_paper_readiness(project)
    pp.readiness_score = readiness["readiness_score"]

    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card("Paper Readiness", f"{readiness['readiness_score']}%", f"{pp.target_format} style")
    with m2:
        render_metric_card("Approved Sections", readiness["approved_sections"], f"{readiness['total_words']} words")
    with m3:
        render_metric_card("Consistency", f"{readiness['consistency_score']}/100", "References OK" if readiness["references_ready"] else "Check Refs")

    st.markdown(f"#### {pp.target_format} Author Guideline Checklist")
    st.dataframe(pd.DataFrame(readiness["guidelines"]), use_container_width=True)

    # Export buttons (DOCX, PDF, LaTeX, BibTeX)
    st.markdown("#### Export Camera-Ready Manuscript Package")
    e1, e2, e3, e4 = st.columns(4)
    base_name = sanitize_filename(project.proposal.title or project.title)
    with e1:
        st.download_button(
            "Download DOCX",
            data=export_docx_bytes(project),
            file_name=f"{base_name}_{pp.target_format}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key="s12_dl_docx",
        )
    with e2:
        st.download_button(
            "Download PDF",
            data=export_pdf_bytes(project),
            file_name=f"{base_name}_{pp.target_format}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="s12_dl_pdf",
        )
    with e3:
        st.download_button(
            "Download LaTeX (.tex)",
            data=export_latex(project),
            file_name=f"{base_name}_{pp.target_format}.tex",
            mime="text/x-tex",
            use_container_width=True,
            key="s12_dl_tex",
        )
    with e4:
        st.download_button(
            "Download BibTeX (.bib)",
            data=export_bibtex(project),
            file_name=f"{base_name}.bib",
            mime="application/x-bibtex",
            use_container_width=True,
            key="s12_dl_bib",
        )

    with st.expander("Target Journal & Venue Discovery (Live OpenAlex Venues)", expanded=False):
        j_query = st.text_input("Search Venues by Domain/Topic", value=project.domain, key="s12_j_query")
        if st.button("Find Matching Journals / Conferences", key="s12_j_btn"):
            journals, err = discover_target_journals(j_query)
            if err:
                st.warning(err)
            else:
                pp.journal_recommendations = journals
        if pp.journal_recommendations:
            st.dataframe(pd.DataFrame(pp.journal_recommendations), use_container_width=True)

    if st.button("Mark Final Paper Complete (Stage 12)", type="primary", key="s12_done"):
        _mark_stage_done(project, "final_paper")
        st.success("Congratulations! Stage 12 completed and preserved.")
