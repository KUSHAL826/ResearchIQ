"""RESEARCHIQ 2.0 — Intelligence Hub UI (Unified KG, Evidence, Landscape, Architects, Experiment & Manuscript Intel)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core.config import MANUSCRIPT_SECTIONS
from core.llm_client import GeminiClient, get_llm_client
from core.schemas import ExperimentRecord, ResearchProject, utc_now_iso
from core.vector_store import SessionVectorStore
from frontend.components import (
    render_metric_card,
    render_project_status_bar,
    render_section_header,
)
from frontend.visualization_components import PLOTLY_LAYOUT_BASE
from paper_studio.results_analyzer import load_project_dataset_df
from research.cross_domain import COMMON_DOMAINS
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
    NODE_PALETTE,
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


OS_MODULES = [
    "1. Unified Research Knowledge Graph",
    "2. Evidence Intelligence",
    "3. Research Landscape Map",
    "4. Contradiction Intelligence",
    "5. Gap Validation Engine",
    "6. Research Problem Architect",
    "7. Methodology Architect",
    "8. Experiment Planning & Analysis Engine",
    "9. Manuscript & Reproducibility Intelligence",
    "10. Publication Strategy & Suitability",
    "11. Project Research Workflow Assistant",
]


def render_research_os_hub(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    assets_cache: dict[str, bytes],
) -> None:
    """Render RESEARCHIQ 2.0 Intelligence Hub cleanly without altering existing workspaces."""
    render_project_status_bar(project, last_autosave_ts=st.session_state.get("last_autosave_ts", ""))
    llm_client = get_llm_client(st.session_state.get("gemini_api_key"))

    render_section_header(
        "RESEARCHIQ 2.0 — Research Operating System Hub",
        "Unified project intelligence connecting papers, evidence passages, landscape maps, methodology design, ROC/PR analytics, and reproducibility audits.",
        badge="RESEARCH OS 2.0",
    )

    default_idx = int(st.session_state.get("os_hub_module_idx", 0))
    selected = st.selectbox(
        "Select Research OS 2.0 Intelligence Module",
        OS_MODULES,
        index=min(default_idx, len(OS_MODULES) - 1),
        key="os2_module_select",
    )
    st.session_state["os_hub_module_idx"] = OS_MODULES.index(selected)

    if selected.startswith("1."):
        _ui_unified_kg(project)
    elif selected.startswith("2."):
        _ui_evidence_intel(project, vector_store, llm_client)
    elif selected.startswith("3."):
        _ui_landscape_map(project)
    elif selected.startswith("4."):
        _ui_contradictions(project, llm_client)
    elif selected.startswith("5."):
        _ui_gap_validator(project, vector_store, llm_client)
    elif selected.startswith("6."):
        _ui_problem_architect(project, llm_client)
    elif selected.startswith("7."):
        _ui_methodology_architect(project, llm_client)
    elif selected.startswith("8."):
        _ui_experiment_engine(project, assets_cache)
    elif selected.startswith("9."):
        _ui_manuscript_intel(project, vector_store, llm_client)
    elif selected.startswith("10."):
        _ui_publication_strategy(project)
    else:
        _ui_workflow_assistant(project, vector_store, llm_client)


def _ui_unified_kg(project: ResearchProject) -> None:
    st.markdown("### Unified Research Knowledge Graph")
    st.caption("Green solid edges = Source Evidence; Violet dashed edges = AI-Inferred relationships.")
    G = build_unified_knowledge_graph(project)
    summary = summarize_unified_graph(G)

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Knowledge Entities", summary["nodes"], "Papers, methods, gaps, metrics")
    with c2:
        render_metric_card("Relationships", summary["edges"], f"{summary['edge_evidence_types'].get('Source Evidence', 0)} source-backed")
    with c3:
        render_metric_card("AI-Inferred Links", summary["edge_evidence_types"].get("AI-Inferred", 0), "Explicitly tagged")

    all_types = sorted(summary["node_types"].keys()) or list(NODE_PALETTE.keys())
    fc1, fc2 = st.columns([1.6, 1.0])
    with fc1:
        picked_types = st.multiselect("Filter Entity Types", all_types, default=all_types, key="os2_kg_types")
    with fc2:
        search_q = st.text_input("Highlight Node", placeholder="Type paper, algorithm, gap...", key="os2_kg_search")

    if summary["nodes"] == 0:
        st.info("Add papers, gaps, or studio objectives to populate the Unified Research Knowledge Graph.")
        return

    html_graph = render_unified_kg_pyvis(G, filter_types=picked_types, search_query=search_q, height_px=520)
    components.html(html_graph, height=540)

    if summary["top_hubs"]:
        st.markdown("#### Most Connected Research Entities")
        st.dataframe(pd.DataFrame(summary["top_hubs"]), use_container_width=True)


def _ui_evidence_intel(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Centralized Research Evidence Intelligence")
    query = st.text_input("Query Evidence Across Session PDFs", value=project.selected_topic or project.domain, key="os2_ev_query")
    if st.button("Aggregate & Classify Traceable Evidence", type="primary", key="os2_ev_btn") or not project.evidence_registry:
        collect_project_evidence(project, vector_store=vector_store, query=query, llm_client=llm_client)
        project.unsaved_changes = True

    items = project.evidence_registry
    if not items:
        st.info("Upload PDFs or run Gap/Claim Verification to populate traceable evidence.")
        return

    counts: dict[str, int] = {}
    for it in items:
        counts[it.classification] = counts.get(it.classification, 0) + 1
    cdf = pd.DataFrame(list(counts.items()), columns=["Classification", "Count"])
    fig = px.bar(cdf, x="Classification", y="Count", color="Classification", title="Evidence Classification Distribution")
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=290)
    st.plotly_chart(fig, use_container_width=True)

    for ev in items[:25]:
        with st.expander(f"[{ev.classification}] {ev.source_paper[:48]} ({ev.page_number}) — {ev.statement[:65]}", expanded=False):
            st.caption(f"Section: {ev.section} | Paper ID: {ev.paper_id or 'N/A'} | Retrieval Score: {ev.retrieval_score}")
            st.write(f"**Statement:** {ev.statement}")
            st.code(ev.passage or "No verbatim passage stored.", language="text")


def _ui_landscape_map(project: ResearchProject) -> None:
    st.markdown("### Research Landscape Mapping")
    c1, c2, c3 = st.columns(3)
    with c1:
        ymin = st.number_input("Min Year", min_value=1990, max_value=2030, value=2015, key="os2_ls_ymin")
    with c2:
        ymax = st.number_input("Max Year", min_value=1990, max_value=2030, value=2027, key="os2_ls_ymax")
    with c3:
        meth_f = st.text_input("Filter by Methodology Keyword", value="", key="os2_ls_meth")
    src_f = st.multiselect("Sources", ["uploaded", "openalex", "arxiv", "semantic_scholar"], default=[], key="os2_ls_src")

    ls = map_research_landscape(project, year_min=int(ymin), year_max=int(ymax), source_filter=src_f, methodology_filter=meth_f)

    ca, cb = st.columns(2)
    with ca:
        if ls["dominant_themes"]:
            df_t = pd.DataFrame(list(ls["dominant_themes"].items()), columns=["Theme", "Count"])
            fig_t = px.bar(df_t, x="Count", y="Theme", orientation="h", title="Dominant Research Themes", color_discrete_sequence=["#6D28D9"])
            fig_t.update_layout(**PLOTLY_LAYOUT_BASE, height=320)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No theme tags available for the active filter.")
    with cb:
        if ls["frequent_algorithms"]:
            df_a = pd.DataFrame(list(ls["frequent_algorithms"].items()), columns=["Algorithm", "Count"])
            fig_a = px.bar(df_a, x="Algorithm", y="Count", title="Frequently Used Algorithms", color_discrete_sequence=["#0284C7"])
            fig_a.update_layout(**PLOTLY_LAYOUT_BASE, height=320)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("No algorithm occurrences matched.")

    cc, cd = st.columns(2)
    with cc:
        if ls["popular_datasets"]:
            st.markdown("#### Popular Datasets")
            st.dataframe(pd.DataFrame(list(ls["popular_datasets"].items()), columns=["Dataset", "Mentions"]), use_container_width=True)
        if ls["emerging_intersections"]:
            st.markdown("#### Emerging Topic Intersections")
            st.dataframe(pd.DataFrame(list(ls["emerging_intersections"].items()), columns=["Intersection", "Papers"]), use_container_width=True)
    with cd:
        if ls["publication_activity"]:
            df_y = pd.DataFrame(list(ls["publication_activity"].items()), columns=["Year", "Publications"])
            fig_y = px.area(df_y, x="Year", y="Publications", title="Filtered Publication Activity", markers=True, color_discrete_sequence=["#059669"])
            fig_y.update_layout(**PLOTLY_LAYOUT_BASE, height=300)
            st.plotly_chart(fig_y, use_container_width=True)
        if ls["identified_limitations"]:
            st.markdown("#### Identified Limitations Across Papers")
            for lim in ls["identified_limitations"][:6]:
                st.caption(f"• {lim}")


def _ui_contradictions(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Research Contradiction Intelligence")
    st.caption("Compares findings alongside experimental context (datasets & metrics) so cross-dataset differences are not mislabelled as contradictions.")
    if st.button("Analyze Cross-Paper Contradictions & Context", type="primary", key="os2_contra_btn"):
        analyze_research_contradictions(project, llm_client=llm_client)
        project.unsaved_changes = True

    items = project.contradiction_analyses
    if not items:
        st.info("Analyse at least 2 papers (or fetch abstracts) and click above to inspect context-checked findings.")
        return

    for it in items:
        with st.container(border=True):
            st.markdown(f"**[{it.contradiction_type}] {it.topic}**")
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**{it.paper_a}** (`{it.context_a}`)")
                st.write(it.claim_a)
            with cb:
                st.markdown(f"**{it.paper_b}** (`{it.context_b}`)")
                st.write(it.claim_b)
            st.info(it.explanation)


def _ui_gap_validator(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Research Gap Validation Engine")
    if st.button("Validate All Identified Gaps", type="primary", key="os2_gv_btn"):
        validate_research_gaps(project, vector_store=vector_store, llm_client=llm_client)
        project.unsaved_changes = True

    reps = project.gap_validation_reports
    if not reps:
        st.info("Detect research gaps in Workspace A (Feature 2) first, then run validation here.")
        return

    for r in reps:
        with st.expander(f"[{r.classification.upper()} | Feasibility: {r.technical_feasibility} | Complexity: {r.complexity}] {r.gap_title}", expanded=False):
            st.write(f"**Reasoning:** {r.reasoning}")
            st.write(f"**Potential Contribution:** {r.potential_contribution}")
            st.write("**Supporting Literature:**", ", ".join(r.supporting_literature) or "None")
            st.write("**Existing Approaches:**", ", ".join(r.existing_approaches) or "None")
            st.write("**Required Datasets:**", ", ".join(r.required_datasets))
            st.write("**Evaluation Possibilities:**", ", ".join(r.evaluation_possibilities))
            st.warning("**Uncertainties:** " + "; ".join(r.uncertainties))
            st.caption("**Further Searches Required:** " + "; ".join(r.further_searches_required))


def _ui_problem_architect(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Research Problem Architect")
    c1, c2 = st.columns(2)
    with c1:
        dom = st.selectbox(
            "Domain",
            COMMON_DOMAINS,
            index=COMMON_DOMAINS.index(project.domain) if project.domain in COMMON_DOMAINS else 0,
            key="os2_pa_dom",
        )
    with c2:
        interest = st.text_input("Research Interest", value=project.selected_topic or "", placeholder="e.g., Clinical RAG verification", key="os2_pa_int")

    if st.button("Generate & Compare Problem Candidates", type="primary", key="os2_pa_btn"):
        generate_problem_candidates(dom, interest, project, llm_client=llm_client)
        project.unsaved_changes = True

    cands = project.problem_candidates
    if not cands:
        st.info("Enter a domain and research interest to compare structured research problem candidates.")
        return

    cdf = pd.DataFrame(
        [
            {
                "Candidate": c.title[:36],
                "Feasibility": c.feasibility_score,
                "Potential Contribution": c.contribution_score,
            }
            for c in cands
        ]
    )
    fig = px.scatter(cdf, x="Feasibility", y="Potential Contribution", hover_name="Candidate", title="Problem Candidate Comparison")
    fig.update_traces(marker=dict(size=15, color="#6D28D9"))
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=280)
    st.plotly_chart(fig, use_container_width=True)

    for idx, c in enumerate(cands):
        with st.container(border=True):
            st.markdown(f"**{c.title}** (Feasibility: {c.feasibility_score}/10 | Contribution: {c.contribution_score}/10)")
            st.write(c.problem_statement)
            st.caption(f"Methodology: {c.suggested_methodology} | Evaluation: {c.evaluation_strategy}")
            if st.button("Select This Problem Candidate → Stage 2", key=f"os2_pa_pick_{idx}"):
                select_problem_candidate(project, c)
                project.unsaved_changes = True
                st.success(f"Selected '{c.title}' and populated Stage 2 Problem Statement.")


def _ui_methodology_architect(project: ResearchProject, llm_client: GeminiClient) -> None:
    st.markdown("### Methodology Architect")
    if st.button("Generate Grounded Methodology Recommendations", type="primary", key="os2_ma_btn"):
        recommend_methodologies(project, llm_client=llm_client)
        project.unsaved_changes = True

    recs = project.methodology_recommendations
    if not recs:
        st.info("Click above to generate method, algorithm, dataset, baseline, and evaluation recommendations.")
        return

    for r in recs:
        with st.container(border=True):
            st.markdown(f"**{r.method_name}**")
            st.write(f"**Suitability:** {r.suitability_reason}")
            st.write(f"**Problem Addressed:** {r.problem_addressed}")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Assumptions:**")
                for a in r.assumptions:
                    st.caption(f"• {a}")
                st.markdown("**Recommended Datasets:** " + ", ".join(r.recommended_datasets))
            with cb:
                st.markdown("**Limitations:**")
                for l in r.limitations:
                    st.caption(f"• {l}")
                st.markdown("**Baselines:** " + ", ".join(r.baselines))
            st.caption(f"**Evaluation Plan:** {r.evaluation_plan} | **Metrics:** {', '.join(r.recommended_metrics)}")


def _ui_experiment_engine(project: ResearchProject, assets_cache: dict[str, bytes]) -> None:
    st.markdown("### Experiment Planning & Analysis Engine")
    sync_default_experiment_records(project)

    # Experiment Status Registry (Planned vs Completed vs Simulated vs Actual Measured)
    with st.expander("Experiment Registry (Planned vs Actual Measured)", expanded=True):
        c1, c2, c3 = st.columns([1.8, 1.1, 1.5])
        with c1:
            exp_name = st.text_input("Experiment Name", placeholder="e.g., Ablation w/o Reranker", key="os2_exp_name")
        with c2:
            exp_status = st.selectbox("Status", ["Planned", "Completed", "Simulated Example", "Actual Measured"], key="os2_exp_stat")
        with c3:
            exp_cfg = st.text_input("Configuration", placeholder="seed=42, lr=1e-4", key="os2_exp_cfg")
        if st.button("Log Experiment Record", key="os2_exp_add") and exp_name.strip():
            project.experiment_records.append(ExperimentRecord(name=exp_name.strip(), status=exp_status, configuration=exp_cfg.strip()))  # type: ignore[arg-type]
            project.unsaved_changes = True

        if project.experiment_records:
            st.dataframe(pd.DataFrame([r.model_dump() for r in project.experiment_records]), use_container_width=True)

    st.info(explain_actual_results(project))

    er = project.experiment_results
    if not er.datasets:
        st.caption("Upload a CSV/Excel dataset in Workspace B -> Stage 9 to unlock ROC/PR curves, statistical CI tables, and error analysis here.")
        return

    sel_ds = st.selectbox("Select Measured Dataset", er.datasets, format_func=lambda d: f"{d.filename} ({d.row_count} rows)", key="os2_exp_ds")
    df = load_project_dataset_df(project, sel_ds, assets_cache)
    if df is None or df.empty:
        return

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Statistical CI Analysis
    if cols and num_cols:
        st.markdown("#### Statistical Group Summary (Mean, Std Dev, 95% CI)")
        sc1, sc2 = st.columns(2)
        with sc1:
            gcol = st.selectbox("Group / Model Column", cols, key="os2_stat_g")
        with sc2:
            mcol = st.selectbox("Numeric Metric Column", num_cols, key="os2_stat_m")
        stat_df = compute_statistical_summary(df, gcol, mcol)
        if not stat_df.empty:
            st.dataframe(stat_df, use_container_width=True)

    # ROC & Precision-Recall Curves + Error Analysis
    with st.expander("ROC & Precision-Recall Curves + Misclassified Error Analysis", expanded=False):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            yt_col = st.selectbox("True Label Column", cols, key="os2_roc_yt")
        with rc2:
            score_col = st.selectbox("Score / Probability Column", num_cols or cols, key="os2_roc_sc")
        with rc3:
            pos_lbl = st.text_input("Positive Label Value", value="1", key="os2_roc_pos")

        if st.button("Compute ROC & Precision-Recall Curves", key="os2_roc_btn"):
            try:
                roc_res = compute_roc_and_pr_curves(df, yt_col, score_col, positive_label=pos_lbl)
                st.session_state["os2_roc_data"] = roc_res
            except Exception as exc:
                st.error(str(exc))

        roc_data = st.session_state.get("os2_roc_data")
        if roc_data:
            p1, p2 = st.columns(2)
            with p1:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode="lines+markers", name=f"ROC (AUROC={roc_data['auroc']})", line=dict(color="#6D28D9")))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#94A3B8")))
                fig_roc.update_layout(**PLOTLY_LAYOUT_BASE, title=f"ROC Curve (AUROC = {roc_data['auroc']})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=320)
                st.plotly_chart(fig_roc, use_container_width=True)
            with p2:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=roc_data["recall"], y=roc_data["precision"], mode="lines+markers", name=f"PR (AUPRC={roc_data['auprc']})", line=dict(color="#059669")))
                fig_pr.update_layout(**PLOTLY_LAYOUT_BASE, title=f"Precision-Recall Curve (AUPRC = {roc_data['auprc']})", xaxis_title="Recall", yaxis_title="Precision", height=320)
                st.plotly_chart(fig_pr, use_container_width=True)

        if len(cols) >= 2:
            yp_err_col = st.selectbox("Predicted Class Column (for Error Analysis)", cols, index=min(1, len(cols) - 1), key="os2_err_yp")
            err_df = extract_error_analysis_rows(df, yt_col, yp_err_col)
            st.caption(f"Misclassified rows detected: {len(err_df)}")
            if not err_df.empty:
                st.dataframe(err_df, use_container_width=True)


def _ui_manuscript_intel(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Manuscript & Reproducibility Intelligence")
    t1, t2, t3, t4 = st.tabs(["A. Writing Assistant", "B. Claim-to-Evidence Map", "C. Completeness & Reproducibility", "D. Consistency Graph"])

    with t1:
        sec_name = st.selectbox("Inspect Section", MANUSCRIPT_SECTIONS, key="os2_wa_sec")
        sec_obj = project.manuscript_sections.get(sec_name)
        text = sec_obj.content if sec_obj else ""
        if st.button("Analyze Academic Writing Quality", type="primary", key="os2_wa_btn"):
            st.session_state["os2_wa_res"] = analyze_writing_quality(sec_name, text, llm_client=llm_client)
        wa = st.session_state.get("os2_wa_res")
        if wa:
            st.caption(f"Words: {wa['word_count']} | Sentences: {wa['sentence_count']} | Repeated terms: {', '.join(wa['repeated_words']) or 'None'}")
            for note in wa["tone_notes"]:
                st.info(note)
            if wa["unsupported_statements"]:
                st.warning("Statements lacking citations or quantitative support:")
                for u in wa["unsupported_statements"]:
                    st.caption(f"• {u}")
            if wa["suggested_revision"]:
                st.text_area("Suggested Non-Destructive Revision (Approved sections are never auto-modified)", value=wa["suggested_revision"], height=200, key="os2_wa_rev")

    with t2:
        if st.button("Map Manuscript Claims to Evidence", type="primary", key="os2_cm_btn"):
            map_claims_to_evidence(project, vector_store, llm_client=llm_client)
            project.unsaved_changes = True
        if project.claim_verifications:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Section": c.section_name,
                            "Claim": c.claim_text[:75],
                            "Source": c.cited_source[:45],
                            "Page": c.page_reference,
                            "Status": c.support_status,
                        }
                        for c in project.claim_verifications
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.info("Click above to scan manuscript sentences containing citations.")

    with t3:
        comp_rows = analyze_manuscript_completeness(project)
        repro = check_research_reproducibility(project)
        ca, cb = st.columns(2)
        with ca:
            cdf = pd.DataFrame(comp_rows)
            fig_c = px.bar(cdf, x="Score", y="Dimension", orientation="h", color="Status", title="10-Point Manuscript Completeness", color_discrete_map={"Complete": "#059669", "Needs Attention": "#D97706"})
            fig_c.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
            st.plotly_chart(fig_c, use_container_width=True)
        with cb:
            render_metric_card("Reproducibility Score", f"{repro.score}%", "8-criterion checklist")
            st.dataframe(pd.DataFrame([i.model_dump() for i in repro.items]), use_container_width=True)

    with t4:
        ac = run_academic_consistency_intelligence(project)
        render_metric_card("Academic Consistency Score", f"{ac['overall_score']}/100", "RQs -> Objectives -> Methodology -> Results")
        if ac["rq_to_objective_map"]:
            st.dataframe(pd.DataFrame(ac["rq_to_objective_map"]), use_container_width=True)
        if ac["objective_to_result_map"]:
            st.dataframe(pd.DataFrame(ac["objective_to_result_map"]), use_container_width=True)


def _ui_publication_strategy(project: ResearchProject) -> None:
    st.markdown("### Publication Strategy & Suitability Intelligence")
    st.warning("Suitability reports rely on OpenAlex venue metadata and do NOT guarantee acceptance, indexing status, or venue legitimacy. Always verify requirements on the official publisher website.")
    q = st.text_input("Topic / Domain Query for Venues", value=project.selected_topic or project.domain, key="os2_pub_q")
    if st.button("Generate Venue Suitability Report", type="primary", key="os2_pub_btn"):
        _, err = build_publication_suitability_report(project, query_override=q)
        if err:
            st.warning(err)
        else:
            project.unsaved_changes = True

    if project.publication_suitability:
        st.dataframe(pd.DataFrame([v.model_dump() for v in project.publication_suitability]), use_container_width=True)


def _ui_workflow_assistant(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> None:
    st.markdown("### Contextual Project Research Workflow Assistant (LangGraph)")
    st.caption("Diagnoses your active project state, evidence coverage, gaps, methodology justification, experiments, and claims without altering approved content.")

    presets = [
        "Which research gaps have the strongest evidence?",
        "Which papers use similar methodologies?",
        "What is missing from my literature review?",
        "Have I justified my selected research method?",
        "Which experiments are still pending?",
        "Do my results support my research objectives?",
        "Which manuscript claims lack evidence?",
    ]
    pcols = st.columns(2)
    chosen_preset = ""
    for idx, pq in enumerate(presets):
        with pcols[idx % 2]:
            if st.button(pq, key=f"os2_wa_preset_{idx}", use_container_width=True):
                chosen_preset = pq

    custom_q = st.text_input("Or ask a project-specific diagnostic question", value=chosen_preset, key="os2_wa_q")
    run_q = chosen_preset or (custom_q if st.button("Analyze Project State", type="primary", key="os2_wa_run") else "")

    if run_q.strip():
        res = ask_project_workflow_assistant(project, run_q.strip(), vector_store=vector_store, llm_client=llm_client)
        project.os_assistant_history.insert(0, {"timestamp": utc_now_iso(), "question": run_q.strip(), "answer": res["answer"]})
        project.os_assistant_history = project.os_assistant_history[:12]
        project.unsaved_changes = True

    for entry in project.os_assistant_history[:5]:
        with st.container(border=True):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(entry["answer"])
