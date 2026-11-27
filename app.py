"""RESEARCHIQ — Visual-First Autonomous Research Development Platform (Streamlit Entrypoint)."""

from __future__ import annotations

import streamlit as st

from core.config import APP_NAME
from core.project_manager import (
    create_empty_project,
    ensure_project_sections,
    session_autosave_backup,
)
from core.schemas import ResearchProject
from core.vector_store import SessionVectorStore
from frontend.animation import inject_theme_css
from frontend.dashboard import render_dashboard
from frontend.paper_studio import render_paper_studio
from frontend.research_workspace import render_research_workspace
from frontend.sidebar import render_sidebar


def _init_session_state() -> None:
    if "projects" not in st.session_state:
        initial = create_empty_project(title="Untitled Research Project", domain="Artificial Intelligence")
        st.session_state["projects"] = {initial.project_id: initial}
        st.session_state["active_project_id"] = initial.project_id
    if "vector_stores" not in st.session_state:
        st.session_state["vector_stores"] = {}
    if "assets_cache" not in st.session_state:
        st.session_state["assets_cache"] = {}
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "Command Dashboard"
    if "autosave_digests" not in st.session_state:
        st.session_state["autosave_digests"] = {}


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} — Autonomous Research Studio",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme_css()
    _init_session_state()

    projects: dict[str, ResearchProject] = st.session_state["projects"]
    active_id: str = st.session_state["active_project_id"]
    if active_id not in projects:
        active_id = next(iter(projects.keys()))
        st.session_state["active_project_id"] = active_id

    active_project = ensure_project_sections(projects[active_id])
    vector_stores: dict[str, SessionVectorStore] = st.session_state["vector_stores"]
    if active_id not in vector_stores:
        vector_stores[active_id] = SessionVectorStore(active_id)

    assets_cache: dict[str, dict[str, bytes]] = st.session_state["assets_cache"]
    if active_id not in assets_cache:
        assets_cache[active_id] = {}

    render_sidebar(projects, active_project, vector_stores, assets_cache)

    nav = st.session_state.get("active_nav", "Command Dashboard")
    if nav == "Command Dashboard":
        render_dashboard(projects, active_project, vector_stores, assets_cache)
    elif nav == "Workspace A — Research Intelligence":
        render_research_workspace(active_project, vector_stores[active_id], assets_cache[active_id])
    else:
        render_paper_studio(active_project, vector_stores[active_id], assets_cache[active_id])

    # Session-scoped automatic backup (writes only when project digest actually changed)
    digests: dict[str, str] = st.session_state["autosave_digests"]
    wrote, new_digest, ts = session_autosave_backup(active_project, last_digest=digests.get(active_id))
    digests[active_id] = new_digest
    if wrote and ts:
        st.session_state["last_autosave_ts"] = ts


if __name__ == "__main__":
    main()
