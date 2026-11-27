"""Main Command Dashboard: Welcome Hero, Live Session Analytics, Clickable 12-Stage Workflow & Project Deck."""

from __future__ import annotations

import html
import streamlit as st

from core.config import PRIVACY_DISCLOSURE, WORKFLOW_STAGES
from core.llm_client import get_llm_client
from core.project_manager import (
    cleanup_project_temp_dir,
    create_empty_project,
    export_project_to_riq_bytes,
    sanitize_filename,
    validate_and_import_riq_bytes,
)
from core.schemas import ResearchProject, utc_now_iso
from core.vector_store import SessionVectorStore
from frontend.components import (
    render_metric_card,
    render_project_status_bar,
    render_section_header,
)
from paper_studio.manuscript_manager import compute_manuscript_completion
from research.systematic_review import compute_prisma_counts


def render_dashboard(
    projects: dict[str, ResearchProject],
    active_project: ResearchProject,
    vector_stores: dict[str, SessionVectorStore],
    assets_cache: dict[str, dict[str, bytes]],
) -> None:
    """Render the RESEARCHIQ visual Command Dashboard."""
    render_project_status_bar(
        active_project,
        last_autosave_ts=st.session_state.get("last_autosave_ts", ""),
    )

    # 1. Welcome Hero Area
    st.markdown(
        """
        <div class="riq-hero">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
                <div style="max-width:740px;">
                    <span class="riq-badge riq-badge-violet">AUTONOMOUS RESEARCH PLATFORM</span>
                    <span class="riq-badge riq-badge-blue">LANGGRAPH + GEMINI + FAISS</span>
                    <h1 style="margin:0.45rem 0 0.35rem 0;font-size:2.05rem;font-weight:800;letter-spacing:-0.03em;color:#F8FAFC;">
                        RESEARCHIQ — Show the Research, Don't Just Describe It.
                    </h1>
                    <p style="margin:0;color:#CBD5E1;font-size:0.96rem;line-height:1.5;">
                        Move from an initial research spark to a validated, publication-ready IEEE, Springer, or ACM manuscript
                        through interactive knowledge graphs, PRISMA screening, evidence-grounded gap detection, and real experimental analytics.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns([1.2, 1.3, 1.4, 1.1])
    with b1:
        if st.button("Start New Research", type="primary", use_container_width=True, key="dash_start_new"):
            st.session_state["show_new_proj_modal"] = True
    with b2:
        if st.button("Upload Research Papers", use_container_width=True, key="dash_upload_papers"):
            st.session_state["active_nav"] = "Workspace A — Research Intelligence"
            st.session_state["workspace_a_tab"] = 0
            st.rerun()
    with b3:
        if st.button("Continue Research Workspace", use_container_width=True, key="dash_continue_ws"):
            st.session_state["active_nav"] = "Workspace B — Paper Studio"
            st.rerun()
    with b4:
        riq_data = export_project_to_riq_bytes(
            active_project,
            asset_bytes_map=assets_cache.get(active_project.project_id, {}),
        )
        st.download_button(
            "Save Project (.riq)",
            data=riq_data,
            file_name=f"{sanitize_filename(active_project.title)}.riq",
            mime="application/zip",
            use_container_width=True,
            key="dash_save_riq",
        )

    # Inline New Research creation panel
    if st.session_state.get("show_new_proj_modal"):
        with st.container(border=True):
            st.markdown("#### Initialize New Research Workspace")
            nc1, nc2 = st.columns(2)
            with nc1:
                new_title = st.text_input("Project Title", value="New AI Research Study", key="dash_new_title")
            with nc2:
                new_domain = st.text_input("Primary Research Domain", value="Artificial Intelligence", key="dash_new_domain")
            ac1, ac2 = st.columns([1, 4])
            with ac1:
                if st.button("Create Workspace", type="primary", key="dash_create_confirm"):
                    np = create_empty_project(title=new_title.strip() or "Untitled Research Project", domain=new_domain.strip() or "Artificial Intelligence")
                    projects[np.project_id] = np
                    vector_stores[np.project_id] = SessionVectorStore(np.project_id)
                    assets_cache[np.project_id] = {}
                    st.session_state["active_project_id"] = np.project_id
                    st.session_state["show_new_proj_modal"] = False
                    st.session_state["active_nav"] = "Workspace B — Paper Studio"
                    st.rerun()
            with ac2:
                if st.button("Cancel", key="dash_create_cancel"):
                    st.session_state["show_new_proj_modal"] = False
                    st.rerun()

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    # 2. Live Research Analytics (Strictly computed from session data)
    prisma = compute_prisma_counts(active_project.discovered_works, active_project.parsed_papers)
    manuscript_comp = compute_manuscript_completion(active_project)
    topics_explored = len(
        {
            *(t.title for t in active_project.topic_ideas),
            *(f for w in active_project.discovered_works for f in w.fields_of_study),
            *(q for q in active_project.trend_queries),
        }
    )
    lit_progress_pct = (
        round((prisma.studies_included / max(1, prisma.records_screened)) * 100.0, 1)
        if prisma.records_screened > 0
        else 0.0
    )
    stage_label = next((s["label"] for s in WORKFLOW_STAGES if s["id"] == active_project.active_stage), active_project.active_stage.upper())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_metric_card("Papers Analysed", len(active_project.parsed_papers), f"{len(active_project.paper_analyses)} deep profiles")
    with m2:
        render_metric_card("Gaps Identified", len(active_project.research_gaps), "Explicit & inferred")
    with m3:
        render_metric_card("Topics Explored", topics_explored, f"{len(active_project.discovered_works)} works fetched")
    with m4:
        render_metric_card("Lit Review Progress", f"{lit_progress_pct}%", f"{prisma.studies_included} included / {prisma.records_screened} screened")
    with m5:
        render_metric_card("Manuscript Completion", f"{manuscript_comp['completion_pct']}%", f"{manuscript_comp['approved_count']}/12 approved")
    with m6:
        render_metric_card("Workflow Stage", stage_label, f"{len(active_project.completed_stages)}/12 completed")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # 3. Interactive 12-Stage Research Workflow Visualization
    render_section_header(
        "Interactive Research Workflow Pipeline",
        "Click any workflow node to jump directly to its corresponding stage in Research Paper Studio.",
        badge="12 STAGES",
    )

    # Visual animated status strip + clickable nodes (2 rows of 6)
    for row_start in (0, 6):
        cols = st.columns(6)
        for offset in range(6):
            idx = row_start + offset
            stg = WORKFLOW_STAGES[idx]
            is_done = stg["id"] in active_project.completed_stages
            is_active = stg["id"] == active_project.active_stage
            state_cls = "completed" if is_done else ("active" if is_active else "")
            status_txt = "COMPLETED" if is_done else ("ACTIVE" if is_active else "PENDING")
            status_color = "#10B981" if is_done else ("#A78BFA" if is_active else "#64748B")
            arrow = " →" if idx < 11 else ""
            with cols[offset]:
                st.markdown(
                    f"""
                    <div class="riq-workflow-step {state_cls}">
                        <div style="font-size:0.66rem;color:{status_color};font-weight:700;letter-spacing:0.05em;">
                            {idx + 1}. {status_txt}{arrow}
                        </div>
                        <div style="font-size:0.78rem;font-weight:600;color:#F3F5FA;margin-top:2px;">
                            {html.escape(stg['label'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {stg['label'].title()}", key=f"wf_node_{stg['id']}", use_container_width=True):
                    active_project.active_stage = stg["id"]
                    active_project.updated_at = utc_now_iso()
                    st.session_state["studio_stage_num"] = stg["studio_stage"]
                    st.session_state["active_nav"] = "Workspace B — Paper Studio"
                    st.rerun()

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # 4. Portable Project Management Dashboard (.riq Import / Export / Multi-Project / Clear)
    render_section_header(
        "Portable Project Management (.riq)",
        "Export portable .riq project archives, import work across computers, switch independent projects, or clear session data.",
        badge="ZERO-LOGIN PORTABLE STORAGE",
    )

    pc1, pc2, pc3 = st.columns([1.2, 1.3, 1.1])
    with pc1:
        with st.container(border=True):
            st.markdown("**Project Details & Export**")
            title_edit = st.text_input("Project Title", value=active_project.title, key="dash_proj_title_edit")
            domain_edit = st.text_input("Primary Domain", value=active_project.domain, key="dash_proj_domain_edit")
            if title_edit != active_project.title or domain_edit != active_project.domain:
                active_project.title = title_edit
                active_project.domain = domain_edit
                active_project.unsaved_changes = True
                active_project.updated_at = utc_now_iso()

            export_bytes = export_project_to_riq_bytes(
                active_project,
                asset_bytes_map=assets_cache.get(active_project.project_id, {}),
            )
            st.download_button(
                "Export Complete Project (.riq)",
                data=export_bytes,
                file_name=f"{sanitize_filename(active_project.title)}.riq",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key="dash_export_riq_main",
            )

    with pc2:
        with st.container(border=True):
            st.markdown("**Import Existing Project (.riq)**")
            imp_file = st.file_uploader("Select .riq archive", type=["riq"], key="dash_import_uploader")
            if imp_file is not None:
                imp_mode = st.radio(
                    "Target",
                    ["Open as New Independent Project", "Overwrite Current Project"],
                    key="dash_imp_mode",
                )
                allow_imp = True
                if imp_mode == "Overwrite Current Project":
                    allow_imp = st.checkbox(
                        "Yes, overwrite my active unsaved project",
                        key="dash_imp_overwrite_chk",
                    )
                if st.button("Validate & Restore .riq", disabled=not allow_imp, use_container_width=True, key="dash_imp_btn"):
                    try:
                        restored, restored_assets, warns = validate_and_import_riq_bytes(imp_file.getvalue())
                        if imp_mode == "Overwrite Current Project":
                            old_id = active_project.project_id
                            cleanup_project_temp_dir(old_id)
                            projects.pop(old_id, None)
                            vector_stores.pop(old_id, None)
                            assets_cache.pop(old_id, None)
                        projects[restored.project_id] = restored
                        assets_cache[restored.project_id] = restored_assets
                        vs = SessionVectorStore(restored.project_id)
                        chunks = [c for p in restored.parsed_papers for c in p.chunks]
                        llm = get_llm_client(st.session_state.get("gemini_api_key"))
                        vs.build_from_chunks(chunks, llm_client=llm)
                        vector_stores[restored.project_id] = vs
                        st.session_state["active_project_id"] = restored.project_id
                        for w in warns:
                            st.warning(w)
                        st.success(f"Successfully restored '{restored.title}' at stage '{restored.active_stage}'.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Archive rejected: {exc}")

    with pc3:
        with st.container(border=True):
            st.markdown("**Session Projects & Cleanup**")
            st.caption(f"Active session projects: **{len(projects)}**")
            for pid, p in list(projects.items()):
                marker = "● " if pid == active_project.project_id else "○ "
                st.caption(f"{marker}{p.title[:28]} — {len(p.parsed_papers)} PDFs, {len(p.discovered_works)} refs")

            st.divider()
            clear_confirm = st.checkbox("Confirm clearing active workspace", key="dash_clear_confirm")
            if st.button("Clear Workspace", disabled=not clear_confirm, use_container_width=True, key="dash_clear_btn"):
                old_id = active_project.project_id
                cleanup_project_temp_dir(old_id)
                fresh = create_empty_project()
                projects.pop(old_id, None)
                vector_stores.pop(old_id, None)
                assets_cache.pop(old_id, None)
                projects[fresh.project_id] = fresh
                vector_stores[fresh.project_id] = SessionVectorStore(fresh.project_id)
                assets_cache[fresh.project_id] = {}
                st.session_state["active_project_id"] = fresh.project_id
                st.rerun()

    st.info(PRIVACY_DISCLOSURE)
