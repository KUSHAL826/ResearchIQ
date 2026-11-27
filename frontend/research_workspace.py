"""Workspace A — Research Intelligence: 10 interactive visual modules."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from agents.research_workflow import run_orchestrated_task
from core.config import STANDARD_PAPER_SECTIONS
from core.llm_client import GeminiClient, get_llm_client
from core.project_manager import get_project_temp_dir, sanitize_filename
from core.schemas import ResearchProject, utc_now_iso
from core.vector_store import SessionVectorStore
from document_engine.chunking import process_pdf_bytes
from frontend.animation import render_real_processing_tracker
from frontend.components import (
    render_gap_card,
    render_metric_card,
    render_project_status_bar,
    render_scholarly_work_card,
    render_section_header,
)
from frontend.visualization_components import (
    PLOTLY_LAYOUT_BASE,
    fig_gap_classification_counts,
    fig_gap_feasibility_impact,
    fig_opportunity_radar,
    fig_paper_results_chart,
    fig_paper_structure_tree,
    fig_prisma_flow,
    fig_publication_timeline,
    fig_trends_heatmap,
    render_mermaid_diagram,
)
from research.citation_network import analyze_graph_metrics, build_citation_graph, render_pyvis_html
from research.citation_verifier import extract_claims_from_manuscript, verify_claim_against_evidence
from research.consistency_checker import run_consistency_check
from research.cross_domain import COMMON_DOMAINS, explore_cross_domain
from research.multi_paper import compare_papers
from research.paper_discovery import deduplicate_works, discover_papers
from research.research_gap import build_evidence_to_gap_graph
from research.systematic_review import (
    apply_rule_based_screening,
    compute_prisma_counts,
    populate_literature_matrix_row,
    synthesize_literature_evidence,
)
from research.trend_analyzer import compute_research_trends


def render_research_workspace(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    assets_cache: dict[str, bytes],
) -> None:
    """Render Workspace A with 10 visual research intelligence features."""
    render_project_status_bar(project, last_autosave_ts=st.session_state.get("last_autosave_ts", ""))
    llm_client = get_llm_client(st.session_state.get("gemini_api_key"))

    feature_names = [
        "1. Paper Intelligence",
        "2. Research Gap Detective",
        "3. Multi-Paper Intelligence",
        "4. Paper Discovery",
        "5. Systematic Review (PRISMA)",
        "6. Citation Network Explorer",
        "7. Cross-Domain Finder",
        "8. Trend & Opportunity Radar",
        "9. Citation Claim Verifier",
        "10. Paper Consistency Checker",
    ]
    default_idx = int(st.session_state.get("workspace_a_tab", 0))
    selected_feature = st.selectbox(
        "Research Intelligence Module",
        feature_names,
        index=min(default_idx, len(feature_names) - 1),
        key="ws_a_module_select",
    )
    st.session_state["workspace_a_tab"] = feature_names.index(selected_feature)

    if selected_feature.startswith("1."):
        _render_feature_1_paper_intelligence(project, vector_store, assets_cache, llm_client)
    elif selected_feature.startswith("2."):
        _render_feature_2_gap_detective(project, vector_store, llm_client)
    elif selected_feature.startswith("3."):
        _render_feature_3_multi_paper(project, llm_client)
    elif selected_feature.startswith("4."):
        _render_feature_4_discovery(project, vector_store, llm_client)
    elif selected_feature.startswith("5."):
        _render_feature_5_systematic_review(project, llm_client)
    elif selected_feature.startswith("6."):
        _render_feature_6_citation_network(project)
    elif selected_feature.startswith("7."):
        _render_feature_7_cross_domain(project, llm_client)
    elif selected_feature.startswith("8."):
        _render_feature_8_trends(project)
    elif selected_feature.startswith("9."):
        _render_feature_9_citation_verifier(project, vector_store, llm_client)
    else:
        _render_feature_10_consistency(project)


# ---------------------------------------------------------------------------
# FEATURE 1 — PAPER INTELLIGENCE
# ---------------------------------------------------------------------------

def _render_feature_1_paper_intelligence(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    assets_cache: dict[str, bytes],
    llm_client: GeminiClient,
) -> None:
    render_section_header(
        "Paper Intelligence Explorer",
        "Upload research PDFs to inspect section hierarchies, methodology flowcharts, datasets, results, limitations, and page-level evidence.",
        badge="FEATURE 1",
    )

    uploaded_files = st.file_uploader(
        "Upload one or more research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="f1_pdf_uploader",
    )

    if uploaded_files and st.button("Process & Index Uploaded PDFs", type="primary", key="f1_process_btn"):
        existing_hashes = {p.file_hash for p in project.parsed_papers}
        stage_box = st.empty()
        completed_live: list[str] = []

        def _stage_cb(stg: str) -> None:
            if stg not in completed_live:
                completed_live.append(stg)
            with stage_box.container():
                render_real_processing_tracker(completed_live, active_stage=stg)

        temp_dir = get_project_temp_dir(project.project_id)
        added = 0
        for uf in uploaded_files:
            raw = uf.getvalue()
            try:
                parsed = process_pdf_bytes(raw, uf.name, progress_cb=_stage_cb)
                if parsed.file_hash in existing_hashes:
                    st.info(f"Skipped duplicate PDF '{uf.name}'.")
                    continue
                rel = f"assets/pdfs/{parsed.paper_id}_{sanitize_filename(uf.name)}"
                parsed.asset_relpath = rel
                assets_cache[rel] = raw
                (temp_dir / "pdfs" / Path(rel).name).write_bytes(raw)
                project.parsed_papers.append(parsed)
                existing_hashes.add(parsed.file_hash)
                added += 1
            except Exception as exc:
                st.error(f"Failed to process '{uf.name}': {exc}")

        if project.parsed_papers:
            all_chunks = [c for p in project.parsed_papers for c in p.chunks]
            count, emb_mode = vector_store.build_from_chunks(all_chunks, llm_client=llm_client)
            _stage_cb("Embeddings generated")
            st.toast(f"Indexed {count} chunks into temporary FAISS ({emb_mode}).")

            # Run Paper Intelligence on newly added papers via Master Orchestrator
            for paper in project.parsed_papers[-added:] if added else []:
                out = run_orchestrated_task(
                    project,
                    vector_store,
                    llm_client,
                    task_type="paper_intelligence",
                    query=paper.title,
                    params={"paper": paper},
                    stage_callback=_stage_cb,
                )
                if out.get("result"):
                    project.paper_analyses[paper.paper_id] = out["result"]

            project.unsaved_changes = True
            project.updated_at = utc_now_iso()
            st.success(f"Processed {added} paper(s). Session vector store updated.")

    if not project.parsed_papers:
        st.info("Upload at least one PDF above to launch the Paper Intelligence Explorer.")
        return

    paper_titles = {p.paper_id: f"{p.title[:70]} ({p.num_pages} pp.)" for p in project.parsed_papers}
    sel_pid = st.selectbox("Select Analysed Paper", list(paper_titles.keys()), format_func=lambda k: paper_titles[k], key="f1_sel_paper")
    paper = next(p for p in project.parsed_papers if p.paper_id == sel_pid)

    col_act1, col_act2 = st.columns([1, 3])
    with col_act1:
        if st.button("Re-run Deep Analysis", key="f1_rerun_analysis"):
            out = run_orchestrated_task(
                project,
                vector_store,
                llm_client,
                task_type="paper_intelligence",
                query=paper.title,
                params={"paper": paper},
            )
            if out.get("result"):
                project.paper_analyses[paper.paper_id] = out["result"]
                project.unsaved_changes = True
                st.rerun()
    with col_act2:
        render_real_processing_tracker(paper.processing_stages_completed + (["Embeddings generated", "Analysis performed", "Results generated"] if paper.paper_id in project.paper_analyses else []))

    analysis = project.paper_analyses.get(paper.paper_id)

    # Interactive Document Overview & Structure Tree
    ov1, ov2 = st.columns([1.2, 1.0])
    with ov1:
        st.plotly_chart(fig_paper_structure_tree(paper), use_container_width=True)
    with ov2:
        st.markdown(f"**Document Overview — {paper.title}**")
        st.caption(f"Authors: {', '.join(paper.authors) or 'Not listed in PDF metadata'} | Year: {paper.year or 'N/A'} | DOI: {paper.doi or 'N/A'}")
        if analysis:
            st.markdown(f"**Problem Statement:** {analysis.problem_statement}")
            if analysis.key_contributions:
                st.markdown("**Key Contributions:**")
                for kc in analysis.key_contributions[:4]:
                    st.markdown(f"- {kc}")
        else:
            st.markdown(paper.hierarchical_summary)

    # Interactive Paper Structure Flow: Abstract -> Introduction -> Literature Review -> Methodology -> Experiments -> Results -> Conclusion
    st.markdown("#### Interactive Section-Wise Breakdown (Click a Section)")
    sec_cols = st.columns(len(STANDARD_PAPER_SECTIONS))
    active_sec_name = st.session_state.get("f1_active_sec", STANDARD_PAPER_SECTIONS[0])
    for idx, sname in enumerate(STANDARD_PAPER_SECTIONS):
        with sec_cols[idx]:
            if st.button(sname, key=f"f1_sec_btn_{sname}", use_container_width=True):
                active_sec_name = sname
                st.session_state["f1_active_sec"] = sname

    matching_secs = [s for s in paper.sections if s.canonical_name.lower() == active_sec_name.lower()]
    if matching_secs:
        for ms in matching_secs:
            with st.container(border=True):
                st.markdown(f"**{ms.canonical_name}** (`{ms.raw_heading}`) — Source Pages: **pp. {ms.start_page}-{ms.end_page}**")
                st.write(ms.content[:3500] + ("..." if len(ms.content) > 3500 else ""))
    else:
        st.info(f"Section '{active_sec_name}' was not explicitly separated by heading detection. Inspect raw pages below.")

    if analysis:
        d1, d2 = st.columns(2)
        with d1:
            render_mermaid_diagram(analysis.mermaid_methodology, height=330, title="Methodology Flowchart")
        with d2:
            render_mermaid_diagram(analysis.mermaid_algorithms, height=330, title="Algorithm Relationship Diagram")

        c_ds, c_res = st.columns([1.0, 1.2])
        with c_ds:
            st.markdown("#### Dataset Information Cards")
            if analysis.datasets:
                for ds in analysis.datasets:
                    with st.container(border=True):
                        st.markdown(f"**{ds.name}** (`{ds.page_reference}`)")
                        st.caption(f"Domain: {ds.domain or 'N/A'} | Split/Size: {ds.size_or_split or 'N/A'}")
            else:
                st.caption("No named datasets automatically parsed.")
        with c_res:
            st.plotly_chart(fig_paper_results_chart(analysis), use_container_width=True)

        st.markdown("#### Extracted Limitation Cards")
        if analysis.limitations:
            lcols = st.columns(2)
            for idx, lim in enumerate(analysis.limitations):
                with lcols[idx % 2]:
                    with st.container(border=True):
                        st.markdown(f"**Severity: {lim.severity}** | `{lim.page_reference}`")
                        st.write(lim.description)
        else:
            st.caption("No explicit limitations extracted yet.")

    # Page-Level Evidence Viewer & Tables/Figures
    with st.expander("Page-Level Evidence Viewer, Tables & Figures", expanded=False):
        pnum = st.slider("Inspect Source Page", min_value=1, max_value=max(1, paper.num_pages), value=1, key="f1_page_slider")
        page_obj = next((p for p in paper.pages if p.page_number == pnum), None)
        if page_obj:
            st.text_area(f"Extracted Text — Page {pnum}", value=page_obj.text, height=260, key=f"f1_page_txt_{pnum}")
        if paper.tables:
            st.markdown("**Extracted Tables**")
            for tb in paper.tables[:6]:
                st.caption(f"{tb.caption} (p. {tb.page_number})")
                st.dataframe(pd.DataFrame(tb.rows, columns=tb.headers if len(tb.headers) == len(tb.rows[0]) else None), use_container_width=True)
        if paper.figures:
            st.markdown("**Detected Figures & Equations**")
            for fig_item in paper.figures[:10]:
                st.caption(f"[{fig_item.kind.upper()} | p. {fig_item.page_number}] {fig_item.caption}: {fig_item.context_snippet[:180]}")


# ---------------------------------------------------------------------------
# FEATURE 2 — RESEARCH GAP DETECTIVE
# ---------------------------------------------------------------------------

def _render_feature_2_gap_detective(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    render_section_header(
        "Research Gap Detective",
        "Distinguish author-stated limitations from AI-inferred opportunities with severity, feasibility, and evidence graphs.",
        badge="FEATURE 2",
    )
    if not project.parsed_papers:
        st.info("Upload and process at least one paper in Feature 1 (Paper Intelligence) first.")
        return

    if st.button("Detect & Classify Research Gaps", type="primary", key="f2_detect_btn"):
        out = run_orchestrated_task(
            project,
            vector_store,
            llm_client,
            task_type="gap_detection",
            query=project.domain,
        )
        if out.get("result") is not None:
            project.research_gaps = out["result"]
            project.unsaved_changes = True
            project.updated_at = utc_now_iso()
            st.success(f"Identified {len(project.research_gaps)} research gaps.")

    if not project.research_gaps:
        st.info("Click 'Detect & Classify Research Gaps' to run gap analysis.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_gap_feasibility_impact(project.research_gaps), use_container_width=True)
    with c2:
        st.plotly_chart(fig_gap_classification_counts(project.research_gaps), use_container_width=True)

    # Evidence-to-Gap Graph
    with st.expander("Evidence-to-Gap Relationship Graph", expanded=True):
        gap_graph = build_evidence_to_gap_graph(project.research_gaps)
        components.html(render_pyvis_html(gap_graph, height_px=400), height=420)

    # Gap Limitation Matrix
    matrix_df = pd.DataFrame(
        [
            {
                "Title": g.title,
                "Type": "Explicit" if g.classification == "explicit" else "Inferred",
                "Category": g.category,
                "Publication": g.supporting_publication[:45],
                "Page": g.page_reference,
                "Feasibility": g.feasibility_score,
                "Impact": g.impact_score,
                "Difficulty": g.implementation_difficulty,
            }
            for g in project.research_gaps
        ]
    )
    st.markdown("#### Research Limitation & Opportunity Matrix")
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown("#### Detailed Gap Cards (Author-Stated vs AI-Inferred)")
    filter_type = st.radio("Filter Gaps", ["All", "Explicit Only", "Inferred Only"], horizontal=True, key="f2_filter")
    for gap in project.research_gaps:
        if filter_type == "Explicit Only" and gap.classification != "explicit":
            continue
        if filter_type == "Inferred Only" and gap.classification != "inferred":
            continue
        render_gap_card(gap)


# ---------------------------------------------------------------------------
# FEATURE 3 — MULTI-PAPER INTELLIGENCE
# ---------------------------------------------------------------------------

def _render_feature_3_multi_paper(project: ResearchProject, llm_client: GeminiClient) -> None:
    render_section_header(
        "Multi-Paper Intelligence & Comparison",
        "Compare methodologies, algorithms, datasets, metrics, contradictions, and overlapping gaps across selected papers.",
        badge="FEATURE 3",
    )
    if len(project.parsed_papers) < 2:
        st.info("Upload at least 2 research papers in Feature 1 to unlock side-by-side comparison.")
        return

    paper_map = {p.paper_id: p for p in project.parsed_papers}
    selected_ids = st.multiselect(
        "Select Papers to Compare",
        options=list(paper_map.keys()),
        default=list(paper_map.keys())[: min(4, len(paper_map))],
        format_func=lambda pid: paper_map[pid].title[:65],
        key="f3_sel_papers",
    )
    if len(selected_ids) < 2:
        st.warning("Select at least 2 papers.")
        return

    selected_papers = [paper_map[pid] for pid in selected_ids]
    if st.button("Run Multi-Paper Comparison", type="primary", key="f3_compare_btn") or project.multi_paper_comparison is None:
        comp = compare_papers(selected_papers, project.paper_analyses, project.research_gaps, llm_client=llm_client)
        project.multi_paper_comparison = comp
        project.unsaved_changes = True

    comp = project.multi_paper_comparison
    if not comp:
        return

    st.info(comp.synthesis_summary)

    # Methodology Comparison Matrix
    st.markdown("#### Side-by-Side Methodology, Algorithm & Dataset Matrix")
    st.dataframe(pd.DataFrame(comp.methodology_matrix), use_container_width=True)

    # Performance comparison chart across selected papers
    perf_rows = []
    for pid in selected_ids:
        a = project.paper_analyses.get(pid)
        if a:
            for r in a.results:
                if r.value is not None:
                    perf_rows.append({"Paper": a.paper_title[:30], "Metric": r.metric, "Value": r.value})
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        if perf_rows:
            fig = px.bar(
                pd.DataFrame(perf_rows),
                x="Metric",
                y="Value",
                color="Paper",
                barmode="group",
                title="Reported Performance Comparison",
                color_discrete_sequence=["#8B5CF6", "#38BDF8", "#10B981", "#F59E0B"],
            )
            fig.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics extracted across selected papers yet.")
    with c2:
        st.markdown("#### Common Themes & Gap Overlap")
        if comp.common_themes:
            st.write("Shared Technical Themes:", ", ".join(f"`{t}`" for t in comp.common_themes))
        if comp.gap_overlaps:
            st.dataframe(pd.DataFrame(comp.gap_overlaps), use_container_width=True)
        else:
            st.caption("Run Gap Detective to visualize overlapping gaps across papers.")

    if comp.contradictions:
        st.markdown("#### Detected Contradictions / Divergent Findings")
        for ctr in comp.contradictions:
            with st.container(border=True):
                st.markdown(f"**{ctr.topic}**")
                st.write(f"- **{ctr.paper_a}**: {ctr.claim_a}")
                st.write(f"- **{ctr.paper_b}**: {ctr.claim_b}")
                st.caption(ctr.analysis)


# ---------------------------------------------------------------------------
# FEATURE 4 — RESEARCH PAPER DISCOVERY
# ---------------------------------------------------------------------------

def _render_feature_4_discovery(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    render_section_header(
        "Scholarly Research Discovery",
        "Search real academic metadata from OpenAlex, arXiv, and Semantic Scholar without fabricated counts or links.",
        badge="FEATURE 4",
    )
    qc1, qc2, qc3 = st.columns([2.2, 1.3, 0.8])
    with qc1:
        query = st.text_input("Search Query", value=project.selected_topic or project.domain, key="f4_query")
    with qc2:
        sources = st.multiselect(
            "Academic APIs",
            ["openalex", "arxiv", "semantic_scholar"],
            default=["openalex", "arxiv", "semantic_scholar"],
            key="f4_sources",
        )
    with qc3:
        year_from = st.number_input("From Year", min_value=1990, max_value=2027, value=2020, key="f4_year")

    if st.button("Search Scholarly Literature", type="primary", key="f4_search_btn"):
        out = run_orchestrated_task(
            project,
            vector_store,
            llm_client,
            task_type="discovery",
            query=query,
            params={"sources": sources, "limit": 8, "year_from": int(year_from)},
        )
        res = out.get("result") or {}
        works = res.get("works", [])
        errs = res.get("errors", [])
        for err in errs:
            st.warning(err)
        if works:
            existing_ids = {w.work_id for w in project.discovered_works}
            new_works = [w for w in works if w.work_id not in existing_ids]
            project.discovered_works.extend(new_works)
            deduplicate_works(project.discovered_works)
            if query and query not in project.trend_queries:
                project.trend_queries.append(query)
            project.unsaved_changes = True
            project.updated_at = utc_now_iso()
            st.success(f"Fetched {len(works)} records ({len(new_works)} new added to session).")

    unique_works = [w for w in project.discovered_works if not w.is_duplicate]
    if not unique_works:
        st.info("Run a search above to explore publications, timelines, and topic clusters.")
        return

    v1, v2 = st.columns([1.15, 1.0])
    with v1:
        st.plotly_chart(fig_publication_timeline(unique_works), use_container_width=True)
    with v2:
        # Topic clusters bar chart
        topic_counts: dict[str, int] = {}
        for w in unique_works:
            for f in w.fields_of_study[:3]:
                topic_counts[f] = topic_counts.get(f, 0) + 1
        if topic_counts:
            tdf = pd.DataFrame(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10], columns=["Topic", "Count"])
            fig_t = px.bar(tdf, x="Count", y="Topic", orientation="h", title="Discovered Topic Clusters", color_discrete_sequence=["#8B5CF6"])
            fig_t.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
            st.plotly_chart(fig_t, use_container_width=True)

    st.markdown(f"#### Discovered Publications ({len(unique_works)} unique)")
    for w in unique_works[:20]:
        render_scholarly_work_card(w)


# ---------------------------------------------------------------------------
# FEATURE 5 — SYSTEMATIC LITERATURE REVIEW ASSISTANT (PRISMA)
# ---------------------------------------------------------------------------

def _render_feature_5_systematic_review(project: ResearchProject, llm_client: GeminiClient) -> None:
    render_section_header(
        "Systematic Literature Review & PRISMA Workbench",
        "Live PRISMA workflow calculated strictly from your screening decisions, with structured matrix extraction and synthesis.",
        badge="FEATURE 5",
    )

    prisma = compute_prisma_counts(project.discovered_works, project.parsed_papers)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Identified", prisma.records_identified, f"{prisma.duplicates_removed} duplicates removed")
    with c2:
        render_metric_card("Screened", prisma.records_screened, f"{prisma.records_excluded} excluded at screening")
    with c3:
        render_metric_card("Full-Text Assessed", prisma.full_text_assessed, f"{prisma.full_text_excluded} excluded full-text")
    with c4:
        render_metric_card("Studies Included", prisma.studies_included, f"{prisma.needs_review} review | {prisma.pending_screening} pending")

    tab_prisma, tab_screen, tab_matrix, tab_synth = st.tabs([
        "A. PRISMA Diagram",
        "B. Paper Screening",
        "C. Literature Matrix",
        "D. Evidence Synthesis",
    ])

    with tab_prisma:
        st.plotly_chart(fig_prisma_flow(prisma), use_container_width=True)
        if prisma.exclusion_reasons:
            st.markdown("**Exclusion Reasons Breakdown**")
            st.dataframe(
                pd.DataFrame(list(prisma.exclusion_reasons.items()), columns=["Reason", "Excluded Papers"]),
                use_container_width=True,
            )

    with tab_screen:
        with st.container(border=True):
            st.markdown("**Automated Criteria Pre-Screening**")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                min_y = st.number_input("Minimum Year", min_value=1990, max_value=2027, value=int(project.screening_criteria.get("min_year", 2018)), key="f5_min_year")
            with sc2:
                req_kw = st.text_input("Required Keywords (comma-separated)", value=", ".join(project.screening_criteria.get("required_keywords", [])), key="f5_req_kw")
            with sc3:
                exc_kw = st.text_input("Exclusion Keywords (comma-separated)", value=", ".join(project.screening_criteria.get("exclusion_keywords", [])), key="f5_exc_kw")
            if st.button("Apply Screening Criteria", key="f5_apply_rules"):
                req_list = [x.strip() for x in req_kw.split(",") if x.strip()]
                exc_list = [x.strip() for x in exc_kw.split(",") if x.strip()]
                project.screening_criteria.update({"min_year": int(min_y), "required_keywords": req_list, "exclusion_keywords": exc_list})
                apply_rule_based_screening(project.discovered_works, min_year=int(min_y), required_keywords=req_list, exclusion_keywords=exc_list)
                project.unsaved_changes = True
                st.rerun()

        unique_works = [w for w in project.discovered_works if not w.is_duplicate]
        if not unique_works and not project.parsed_papers:
            st.info("Discover papers in Feature 4 or upload PDFs in Feature 1 to screen them here.")
        if project.parsed_papers:
            st.markdown("##### Uploaded Research PDFs (Require Explicit Screening)")
            for pidx, p in enumerate(project.parsed_papers):
                if p.is_duplicate:
                    continue
                with st.container(border=True):
                    pl, pr = st.columns([2.2, 1.3])
                    with pl:
                        st.markdown(f"**{p.title}** ({p.year or 'n.d.'} | Uploaded PDF, {p.num_pages} pp.)")
                        st.caption(p.hierarchical_summary[:180])
                    with pr:
                        statuses = ["unscreened", "included", "excluded", "needs_review"]
                        p_stat = st.selectbox(
                            "Decision",
                            statuses,
                            index=statuses.index(p.screening_status),
                            key=f"f5_pdf_stat_{p.paper_id}_{pidx}",
                        )
                        p_ft = st.checkbox("Full-text assessed", value=p.full_text_assessed, key=f"f5_pdf_ft_{p.paper_id}_{pidx}")
                        p_reason = st.text_input("Exclusion reason", value=p.exclusion_reason, key=f"f5_pdf_reas_{p.paper_id}_{pidx}")
                        if p_stat != p.screening_status or p_ft != p.full_text_assessed or p_reason != p.exclusion_reason:
                            p.screening_status = p_stat  # type: ignore[assignment]
                            p.full_text_assessed = p_ft
                            p.exclusion_reason = p_reason
                            project.unsaved_changes = True
                            st.rerun()

        for idx, w in enumerate(unique_works[:25]):
            with st.container(border=True):
                top_l, top_r = st.columns([2.2, 1.3])
                with top_l:
                    st.markdown(f"**{w.title}** ({w.year or 'n.d.'} | {w.source})")
                    st.caption(f"{', '.join(w.authors[:3])} — {(w.abstract or '')[:180]}...")
                with top_r:
                    statuses = ["unscreened", "included", "excluded", "needs_review"]
                    new_stat = st.selectbox(
                        "Decision",
                        statuses,
                        index=statuses.index(w.screening_status),
                        key=f"f5_stat_{w.work_id}_{idx}",
                    )
                    ft = st.checkbox("Full-text assessed", value=w.full_text_assessed, key=f"f5_ft_{w.work_id}_{idx}")
                    reason = st.text_input("Exclusion reason", value=w.exclusion_reason, key=f"f5_reas_{w.work_id}_{idx}")
                    if new_stat != w.screening_status or ft != w.full_text_assessed or reason != w.exclusion_reason:
                        w.screening_status = new_stat  # type: ignore[assignment]
                        w.full_text_assessed = ft
                        w.exclusion_reason = reason
                        project.unsaved_changes = True
                        st.rerun()

    with tab_matrix:
        unique_works = [w for w in project.discovered_works if not w.is_duplicate]
        if st.button("Extract Literature Matrix Fields", type="primary", key="f5_extract_matrix"):
            for w in unique_works[:15]:
                if not w.extracted_methodology:
                    populate_literature_matrix_row(w, llm_client=llm_client)
            project.unsaved_changes = True
            st.success("Literature matrix populated.")

        status_filter = st.multiselect(
            "Filter by Screening Status",
            ["included", "needs_review", "unscreened", "excluded"],
            default=["included", "needs_review", "unscreened"],
            key="f5_mat_filter",
        )
        rows = [
            {
                "Author": ", ".join(w.authors[:2]) or "Unknown",
                "Year": w.year or "N/A",
                "Title": w.title[:48],
                "Status": w.screening_status,
                "Problem": w.extracted_problem or "-",
                "Methodology": w.extracted_methodology or "-",
                "Dataset": w.extracted_dataset or "-",
                "Results": w.extracted_results or "-",
                "Limitations": w.extracted_limitations or "-",
                "Contribution": w.extracted_contribution or "-",
            }
            for w in unique_works
            if w.screening_status in status_filter
        ]
        for p in project.parsed_papers:
            if not p.is_duplicate and p.screening_status in status_filter:
                pa = project.paper_analyses.get(p.paper_id)
                rows.append(
                    {
                        "Author": ", ".join(p.authors[:2]) or "Uploaded PDF",
                        "Year": p.year or "N/A",
                        "Title": p.title[:48],
                        "Status": p.screening_status,
                        "Problem": (pa.problem_statement[:120] if pa else p.hierarchical_summary[:120]) or "-",
                        "Methodology": (" -> ".join(pa.methodology_steps[:3]) if pa else "-"),
                        "Dataset": (", ".join(d.name for d in pa.datasets) if pa and pa.datasets else "-"),
                        "Results": ("; ".join(f"{r.metric}: {r.raw_value or r.value}" for r in pa.results[:3]) if pa and pa.results else "-"),
                        "Limitations": ("; ".join(l.description[:80] for l in pa.limitations[:2]) if pa and pa.limitations else "-"),
                        "Contribution": ("; ".join(pa.key_contributions[:2]) if pa and pa.key_contributions else "-"),
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab_synth:
        synth = synthesize_literature_evidence(project.discovered_works)
        st.caption(f"Synthesizing {synth['total_synthesized']} studies.")
        sc_a, sc_b = st.columns(2)
        with sc_a:
            if synth["methodology_distribution"]:
                mdf = pd.DataFrame(list(synth["methodology_distribution"].items()), columns=["Methodology", "Count"])
                fig_m = px.bar(mdf, x="Methodology", y="Count", title="Methodology Distribution", color_discrete_sequence=["#8B5CF6"])
                fig_m.update_layout(**PLOTLY_LAYOUT_BASE, height=320)
                st.plotly_chart(fig_m, use_container_width=True)
            else:
                st.info("Populate the Literature Matrix to view methodology distribution.")
        with sc_b:
            if synth["theme_distribution"]:
                tdf = pd.DataFrame(list(synth["theme_distribution"].items()), columns=["Theme", "Count"])
                fig_th = px.bar(tdf, x="Count", y="Theme", orientation="h", title="Research Theme Clusters", color_discrete_sequence=["#38BDF8"])
                fig_th.update_layout(**PLOTLY_LAYOUT_BASE, height=320)
                st.plotly_chart(fig_th, use_container_width=True)


# ---------------------------------------------------------------------------
# FEATURE 6 — CITATION NETWORK EXPLORER
# ---------------------------------------------------------------------------

def _render_feature_6_citation_network(project: ResearchProject) -> None:
    render_section_header(
        "Citation & Knowledge Network Explorer",
        "Interactive NetworkX + PyVis graph connecting papers, authors, topics, and shared methodologies.",
        badge="FEATURE 6",
    )
    G = build_citation_graph(project.discovered_works, project.parsed_papers, project.paper_analyses)
    metrics = analyze_graph_metrics(G)

    if metrics["nodes"] == 0:
        st.info("Upload papers or discover scholarly works first to construct the citation network.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Graph Nodes", metrics["nodes"], "Papers, authors, topics, refs")
    with c2:
        render_metric_card("Graph Edges", metrics["edges"], "Cites, written_by, shares_methodology")
    with c3:
        render_metric_card("Central Papers Ranked", len(metrics["central_papers"]), "PageRank + Degree Centrality")

    search_q = st.text_input("Highlight / Search Node", placeholder="Type paper, author, or topic...", key="f6_search")
    components.html(render_pyvis_html(G, search_highlight=search_q, height_px=540), height=560)

    col_rank, col_insp = st.columns([1.2, 1.0])
    with col_rank:
        st.markdown("#### Central & Influential Publications")
        st.dataframe(pd.DataFrame(metrics["central_papers"]), use_container_width=True)
    with col_insp:
        st.markdown("#### Node Inspector")
        paper_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") in ("paper", "uploaded_paper")]
        if paper_nodes:
            sel_n = st.selectbox("Inspect Paper Node", paper_nodes, format_func=lambda nid: G.nodes[nid].get("full_title", nid)[:60], key="f6_insp")
            ndata = G.nodes[sel_n]
            st.markdown(f"**{ndata.get('full_title')}**")
            st.caption(f"Year: {ndata.get('year')} | Citations: {ndata.get('citations')} | DOI: {ndata.get('doi')}")
            st.write(ndata.get("summary") or "No abstract/summary available.")


# ---------------------------------------------------------------------------
# FEATURE 7 — CROSS-DOMAIN RESEARCH FINDER
# ---------------------------------------------------------------------------

def _render_feature_7_cross_domain(project: ResearchProject, llm_client: GeminiClient) -> None:
    render_section_header(
        "Cross-Domain Research Finder",
        "Explore interdisciplinary intersections backed by real scholarly literature counts before claiming novelty.",
        badge="FEATURE 7",
    )
    selected_domains = st.multiselect(
        "Select 2 or more Research Domains",
        COMMON_DOMAINS,
        default=["Artificial Intelligence", "Healthcare"],
        key="f7_domains",
    )
    if st.button("Analyze Cross-Domain Intersection", type="primary", key="f7_run_btn"):
        ideas, works, evidence_breakdown, errors = explore_cross_domain(selected_domains, llm_client=llm_client)
        for err in errors:
            st.warning(err)
        project.cross_domain_ideas = ideas
        st.session_state["f7_domain_counts"] = evidence_breakdown
        st.session_state["f7_intersection_works"] = works
        project.unsaved_changes = True

    breakdown = st.session_state.get("f7_domain_counts") or {}
    counts = breakdown.get("counts", breakdown) if isinstance(breakdown, dict) else {}
    if counts:
        if isinstance(breakdown, dict) and breakdown.get("novelty_status"):
            st.warning(breakdown["novelty_status"])
        cdf = pd.DataFrame(list(counts.items()), columns=["Scope", "Matching Publications"])
        c1, c2 = st.columns(2)
        with c1:
            fig_c = px.bar(cdf, x="Scope", y="Matching Publications", title="Domain & Intersection Literature Evidence", color_discrete_sequence=["#8B5CF6"])
            fig_c.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
            st.plotly_chart(fig_c, use_container_width=True)
        with c2:
            if project.cross_domain_ideas:
                idf = pd.DataFrame(
                    [
                        {
                            "Idea": i.title[:35],
                            "Novelty": i.novelty_score,
                            "Feasibility": i.feasibility_score,
                            "Impact": i.impact_score,
                        }
                        for i in project.cross_domain_ideas
                    ]
                )
                fig_sc = px.scatter(
                    idf,
                    x="Feasibility",
                    y="Impact",
                    size="Novelty",
                    hover_name="Idea",
                    title="Cross-Domain Feasibility vs Impact",
                    color_discrete_sequence=["#10B981"],
                )
                fig_sc.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
                st.plotly_chart(fig_sc, use_container_width=True)

        if isinstance(breakdown, dict) and ("existing_research" in breakdown or "related_research" in breakdown):
            e_col, r_col = st.columns(2)
            with e_col:
                st.markdown("**1. Existing Intersection Research Found**")
                for t in breakdown.get("existing_research", []) or ["None matched all selected domains."]:
                    st.caption(f"• {t}")
            with r_col:
                st.markdown("**2. Related Single-Domain Research Found**")
                for t in breakdown.get("related_research", []) or ["None returned."]:
                    st.caption(f"• {t}")
            st.markdown("**3. Potentially Unexplored Intersection Areas**")
            for t in breakdown.get("potentially_unexplored", []):
                st.caption(f"• {t}")

    for idea in project.cross_domain_ideas:
        with st.container(border=True):
            st.markdown(f"**{idea.title}** — `{' + '.join(idea.domains)}`")
            st.write(idea.description)
            st.caption(f"Novelty Evidence: {idea.novelty_evidence}")


# ---------------------------------------------------------------------------
# FEATURE 8 — RESEARCH TREND & OPPORTUNITY RADAR
# ---------------------------------------------------------------------------

def _render_feature_8_trends(project: ResearchProject) -> None:
    render_section_header(
        "Research Trend & Opportunity Radar",
        "Analytics derived strictly from retrieved scholarly metadata in your active session.",
        badge="FEATURE 8",
    )
    trends = compute_research_trends(project.discovered_works)
    if trends["total_works"] == 0:
        st.info("Search and retrieve papers in Feature 4 (Paper Discovery) to power real trend analytics.")
        return

    st.caption(
        f"Data Source: {', '.join(trends['sources']).upper()} | Time Period: {trends['year_range']} | "
        f"Total Works: {trends['total_works']} (Dated: {trends['dated_works']}, Missing Year: {trends['missing_year_count']})"
    )
    if trends.get("small_sample_warning"):
        st.warning(trends["small_sample_warning"])
    st.info(trends["heuristic_disclaimer"])

    c1, c2 = st.columns(2)
    with c1:
        ydf = pd.DataFrame(list(trends["yearly_counts"].items()), columns=["Year", "Publications"])
        fig_y = px.area(ydf, x="Year", y="Publications", title="Publication Growth Over Time", markers=True, color_discrete_sequence=["#8B5CF6"])
        fig_y.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
        st.plotly_chart(fig_y, use_container_width=True)
    with c2:
        st.plotly_chart(fig_opportunity_radar(trends["opportunity_radar"]), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_trends_heatmap(trends["topic_year_matrix"]), use_container_width=True)
    with c4:
        alg_rows = []
        for alg, ymap in trends["algorithm_trends"].items():
            for yr, cnt in ymap.items():
                if yr > 0:
                    alg_rows.append({"Year": yr, "Algorithm": alg, "Mentions": cnt})
        if alg_rows:
            fig_a = px.line(pd.DataFrame(alg_rows), x="Year", y="Mentions", color="Algorithm", markers=True, title="Algorithm Adoption Trends")
            fig_a.update_layout(**PLOTLY_LAYOUT_BASE, height=360)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("No algorithm trend keywords matched across dated abstracts yet.")


# ---------------------------------------------------------------------------
# FEATURE 9 — CITATION CLAIM VERIFIER
# ---------------------------------------------------------------------------

def _render_feature_9_citation_verifier(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    render_section_header(
        "Citation Claim Verifier",
        "Verify claims strictly against indexed PDF passages and retrieved abstracts. DOI existence alone never marks a claim verified.",
        badge="FEATURE 9",
    )

    custom_claim = st.text_area(
        "Verify a Specific Claim (or leave blank to scan all Manuscript sections)",
        placeholder="e.g., Transformer-based retrievers outperform BM25 on multi-hop QA [1].",
        key="f9_claim_input",
    )
    cited_src = st.text_input("Cited Source Marker (optional)", placeholder="e.g., [1] or Vaswani et al., 2017", key="f9_cite_src")

    b1, b2 = st.columns(2)
    refs_sec = project.manuscript_sections.get("References")
    refs_text = refs_sec.content if refs_sec else ""
    with b1:
        if st.button("Verify Single Claim", type="primary", disabled=not custom_claim.strip(), key="f9_verify_one"):
            res = verify_claim_against_evidence(
                claim_text=custom_claim.strip(),
                cited_source=cited_src.strip(),
                section_name="Manual Check",
                vector_store=vector_store,
                discovered_works=project.discovered_works,
                llm_client=llm_client,
                parsed_papers=project.parsed_papers,
                references_text=refs_text,
            )
            project.claim_verifications.insert(0, res)
            project.unsaved_changes = True
    with b2:
        if st.button("Scan & Verify Manuscript Claims", key="f9_verify_ms"):
            extracted = extract_claims_from_manuscript(project.manuscript_sections)
            if not extracted:
                st.warning("No citation-bearing sentences found in drafted manuscript sections yet.")
            else:
                results = [
                    verify_claim_against_evidence(
                        claim_text=clm,
                        cited_source=marker,
                        section_name=sec,
                        vector_store=vector_store,
                        discovered_works=project.discovered_works,
                        llm_client=llm_client,
                        parsed_papers=project.parsed_papers,
                        references_text=refs_text,
                    )
                    for sec, clm, marker in extracted
                ]
                project.claim_verifications = results
                project.unsaved_changes = True

    if not project.claim_verifications:
        st.info("Submit a claim or scan drafted manuscript sections to inspect claim-to-evidence alignment.")
        return

    status_counts: dict[str, int] = {}
    for c in project.claim_verifications:
        status_counts[c.support_status] = status_counts.get(c.support_status, 0) + 1
    cdf = pd.DataFrame(list(status_counts.items()), columns=["Classification", "Claims"])
    fig_v = px.pie(
        cdf,
        names="Classification",
        values="Claims",
        hole=0.5,
        title="Claim Verification Breakdown",
        color="Classification",
        color_discrete_map={
            "Supported": "#10B981",
            "Partially supported": "#F59E0B",
            "Not supported": "#F43F5E",
            "Unable to verify": "#64748B",
        },
    )
    fig_v.update_layout(**PLOTLY_LAYOUT_BASE, height=300)
    st.plotly_chart(fig_v, use_container_width=True)

    for clm in project.claim_verifications:
        with st.expander(f"[{clm.support_status}] ({clm.section_name}) {clm.claim_text[:95]}", expanded=False):
            st.markdown(f"**Cited Source:** {clm.cited_source} | **Page Ref:** {clm.page_reference}")
            st.caption(f"Confidence Indicator: {clm.confidence_score:.2f} ({clm.confidence_label})")
            st.markdown(f"**Reasoning:** {clm.reasoning}")
            if clm.contradiction_note:
                st.warning(f"Contradiction Alert: {clm.contradiction_note}")
            st.code(clm.evidence_passage, language="text")


# ---------------------------------------------------------------------------
# FEATURE 10 — PAPER CONSISTENCY CHECKER
# ---------------------------------------------------------------------------

def _render_feature_10_consistency(project: ResearchProject) -> None:
    render_section_header(
        "Paper Consistency Checker",
        "Audit alignment across Problem Statement, Objectives, Methodology, Experiments, Results, and Conclusion.",
        badge="FEATURE 10",
    )
    if st.button("Run Full Consistency Audit", type="primary", key="f10_run_btn") or project.consistency_report is None:
        project.consistency_report = run_consistency_check(project)
        project.unsaved_changes = True

    rep = project.consistency_report
    if not rep:
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Manuscript Consistency Score", f"{rep.overall_score}/100", "Objective & claim alignment")
    with c2:
        render_metric_card("Open Issues", sum(1 for i in rep.issues if not i.resolved), f"{len(rep.issues)} total flagged")
    with c3:
        render_metric_card("Objectives Tracked", len(rep.objective_to_result_map), "Mapped to experiments & results")

    t1, t2, t3 = st.tabs(["A. Objective-to-Result Map", "B. Methodology-to-Experiment Map", "C. Issues & Fix Actions"])
    with t1:
        if rep.objective_to_result_map:
            st.dataframe(pd.DataFrame(rep.objective_to_result_map), use_container_width=True)
        else:
            st.info("Define objectives in Studio Stage 2 to map them against experiments and results.")
    with t2:
        if rep.methodology_to_experiment_map:
            st.dataframe(pd.DataFrame(rep.methodology_to_experiment_map), use_container_width=True)
        else:
            st.info("Define methodology algorithms in Studio Stage 6 to map them to experiments.")
    with t3:
        if not rep.issues:
            st.success("No consistency issues detected.")
        for idx, iss in enumerate(rep.issues):
            with st.container(border=True):
                ic1, ic2 = st.columns([3, 1])
                with ic1:
                    st.markdown(f"**[{iss.severity}] {iss.category}** (`{iss.source_section}` -> `{iss.target_section}`)")
                    st.write(iss.description)
                    st.caption(f"Recommendation: {iss.recommendation}")
                with ic2:
                    if st.button("Resolved ✓" if iss.resolved else "Mark Fixed", key=f"f10_fix_{iss.issue_id}_{idx}"):
                        iss.resolved = not iss.resolved
                        project.unsaved_changes = True
                        st.rerun()
