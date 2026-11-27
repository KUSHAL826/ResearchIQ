"""Sidebar navigation, multi-project switcher, .riq persistence controls, and Gemini configuration."""

from __future__ import annotations

import streamlit as st

from core.config import APP_NAME, PRIVACY_DISCLOSURE, WORKFLOW_STAGES
from core.llm_client import get_llm_client
from core.project_manager import (
    cleanup_project_temp_dir,
    create_empty_project,
    export_project_to_riq_bytes,
    sanitize_filename,
    validate_and_import_riq_bytes,
)
from core.schemas import ResearchProject
from core.vector_store import SessionVectorStore


def render_sidebar(
    projects: dict[str, ResearchProject],
    active_project: ResearchProject,
    vector_stores: dict[str, SessionVectorStore],
    assets_cache: dict[str, dict[str, bytes]],
) -> None:
    """Render sleek dark navigation sidebar with multi-project & .riq management."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.4rem 0 0.8rem 0;">
                <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.02em;
                     background:linear-gradient(90deg,#F3F5FA,#A78BFA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    RESEARCHIQ
                </div>
                <div style="font-size:0.74rem;color:#94A3B8;">Visual Autonomous Research Studio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Workspace Navigation
        nav_options = [
            "Command Dashboard",
            "Workspace A — Research Intelligence",
            "Workspace B — Paper Studio",
        ]
        current_nav = st.session_state.get("active_nav", nav_options[0])
        if current_nav not in nav_options:
            current_nav = nav_options[0]
        chosen_nav = st.radio(
            "Workspace",
            nav_options,
            index=nav_options.index(current_nav),
            label_visibility="collapsed",
        )
        st.session_state["active_nav"] = chosen_nav

        st.divider()

        # Multi-project Switcher
        st.markdown('<div style="font-size:0.74rem;color:#94A3B8;font-weight:600;margin-bottom:4px;">PROJECTS (SESSION ISOLATED)</div>', unsafe_allow_html=True)
        proj_ids = list(projects.keys())
        selected_pid = st.selectbox(
            "Active Project",
            proj_ids,
            index=proj_ids.index(active_project.project_id),
            format_func=lambda pid: f"{projects[pid].title[:26]} ({len(projects[pid].parsed_papers)} PDFs)",
            label_visibility="collapsed",
        )
        if selected_pid != active_project.project_id:
            st.session_state["active_project_id"] = selected_pid
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("New Project", use_container_width=True, key="sb_new_proj"):
                new_p = create_empty_project(title=f"Research Project #{len(projects) + 1}")
                projects[new_p.project_id] = new_p
                vector_stores[new_p.project_id] = SessionVectorStore(new_p.project_id)
                assets_cache[new_p.project_id] = {}
                st.session_state["active_project_id"] = new_p.project_id
                st.rerun()
        with c2:
            riq_bytes = export_project_to_riq_bytes(
                active_project,
                asset_bytes_map=assets_cache.get(active_project.project_id, {}),
            )
            st.download_button(
                "Save .riq",
                data=riq_bytes,
                file_name=f"{sanitize_filename(active_project.title)}.riq",
                mime="application/zip",
                use_container_width=True,
                key="sb_save_riq",
            )

        # Quick Import .riq in sidebar with overwrite guard
        with st.expander("Import .riq Project", expanded=False):
            uploaded_riq = st.file_uploader("Upload .riq archive", type=["riq"], key="sb_import_riq_file")
            if uploaded_riq is not None:
                mode = st.radio(
                    "Import mode",
                    ["Open as New Project", "Replace Active Project"],
                    key="sb_import_mode",
                )
                confirm_ok = True
                if mode == "Replace Active Project":
                    confirm_ok = st.checkbox(
                        "Confirm overwriting active project",
                        key="sb_confirm_overwrite",
                    )
                if st.button("Restore Project", use_container_width=True, disabled=not confirm_ok, key="sb_restore_btn"):
                    try:
                        restored, restored_assets, warns = validate_and_import_riq_bytes(uploaded_riq.getvalue())
                        if mode == "Replace Active Project":
                            old_id = active_project.project_id
                            cleanup_project_temp_dir(old_id)
                            projects.pop(old_id, None)
                            vector_stores.pop(old_id, None)
                            assets_cache.pop(old_id, None)

                        projects[restored.project_id] = restored
                        assets_cache[restored.project_id] = restored_assets
                        vs = SessionVectorStore(restored.project_id)
                        all_chunks = [c for p in restored.parsed_papers for c in p.chunks]
                        llm = get_llm_client(st.session_state.get("gemini_api_key"))
                        vs.build_from_chunks(all_chunks, llm_client=llm)
                        vector_stores[restored.project_id] = vs
                        st.session_state["active_project_id"] = restored.project_id
                        for w in warns:
                            st.warning(w)
                        st.success(f"Restored '{restored.title}'.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Import failed: {exc}")

        # Clear workspace with confirmation
        with st.expander("Clear Workspace", expanded=False):
            st.caption("Resets the active project and deletes its temporary files.")
            confirm_clear = st.checkbox("Confirm clearing active project", key="sb_confirm_clear")
            if st.button("Clear Active Project", use_container_width=True, disabled=not confirm_clear, key="sb_clear_btn"):
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

        st.divider()

        # Gemini API Key (session-only, never persisted to DB)
        st.markdown('<div style="font-size:0.74rem;color:#94A3B8;font-weight:600;margin-bottom:4px;">GEMINI AI ENGINE</div>', unsafe_allow_html=True)
        api_key_in = st.text_input(
            "Gemini API Key (session-only)",
            type="password",
            value=st.session_state.get("gemini_api_key", ""),
            placeholder="AIza... (optional if in .env)",
            key="sb_api_key_input",
        )
        if api_key_in != st.session_state.get("gemini_api_key", ""):
            st.session_state["gemini_api_key"] = api_key_in

        llm_client = get_llm_client(st.session_state.get("gemini_api_key"))
        if llm_client.is_configured:
            st.markdown('<span class="riq-badge riq-badge-green">Gemini Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="riq-badge riq-badge-amber">Heuristic Mode (No API Key)</span>', unsafe_allow_html=True)

        # Stage indicator
        stage_titles = {s["id"]: s["label"] for s in WORKFLOW_STAGES}
        st.caption(f"Current Stage: **{stage_titles.get(active_project.active_stage, active_project.active_stage)}**")

        with st.expander("Privacy & Data Notice", expanded=False):
            st.caption(PRIVACY_DISCLOSURE)
