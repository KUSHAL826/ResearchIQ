"""Reusable dark SaaS UI components for RESEARCHIQ."""

from __future__ import annotations

import html
import streamlit as st

from core.schemas import ResearchGap, ResearchProject, ScholarlyWork


def render_metric_card(label: str, value: str | int | float, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="riq-metric">
            <div class="riq-metric-label">{html.escape(str(label))}</div>
            <div class="riq-metric-value">{html.escape(str(value))}</div>
            <div class="riq-metric-sub">{html.escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", badge: str = "") -> None:
    badge_html = f'<span class="riq-badge riq-badge-violet">{html.escape(badge)}</span>' if badge else ""
    st.markdown(
        f"""
        <div style="margin-bottom:0.9rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;">
                {badge_html}
                <h2 style="margin:0;font-size:1.35rem;font-weight:700;color:#F3F5FA;">{html.escape(title)}</h2>
            </div>
            <p style="margin:0.25rem 0 0 0;color:#94A3B8;font-size:0.9rem;">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_status_bar(project: ResearchProject, last_autosave_ts: str = "") -> None:
    """Show active project title, stage, unsaved indicator, and temporary vs .riq backup status."""
    save_badge = (
        '<span class="riq-badge riq-badge-amber">Unsaved Changes (Export .riq)</span>'
        if project.unsaved_changes
        else '<span class="riq-badge riq-badge-green">Synced / Exported</span>'
    )
    export_txt = project.last_exported_at[:19].replace("T", " ") + " UTC" if project.last_exported_at else "Never exported (.riq)"
    autosave_txt = (
        f"Session auto-backup: {last_autosave_ts[:19].replace('T', ' ')} UTC (temporary)"
        if last_autosave_ts
        else "Session auto-backup: active (temporary)"
    )
    st.markdown(
        f"""
        <div class="riq-glass" style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:0.75rem;margin-bottom:1rem;">
            <div>
                <span style="font-size:0.72rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.06em;">Active Research Workspace</span>
                <div style="font-size:1.05rem;font-weight:600;color:#F3F5FA;">{html.escape(project.title)}
                    <span class="riq-badge riq-badge-blue" style="margin-left:0.4rem;">Stage: {html.escape(project.active_stage.upper())}</span>
                    {save_badge}
                </div>
            </div>
            <div style="text-align:right;font-size:0.76rem;color:#94A3B8;">
                <div>Last .riq export: <b style="color:#E2E8F0;">{html.escape(export_txt)}</b></div>
                <div>{html.escape(autosave_txt)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gap_card(gap: ResearchGap) -> None:
    """Visually distinguish author-stated (explicit) vs AI-inferred research gaps."""
    is_explicit = gap.classification == "explicit"
    cls_name = "riq-gap-explicit" if is_explicit else "riq-gap-inferred"
    type_badge = (
        '<span class="riq-badge riq-badge-green">AUTHOR-STATED (EXPLICIT)</span>'
        if is_explicit
        else '<span class="riq-badge riq-badge-violet">AI-INFERRED GAP</span>'
    )
    diff_badge = (
        f'<span class="riq-badge riq-badge-amber">Difficulty: {html.escape(gap.implementation_difficulty)}</span>'
    )
    st.markdown(
        f"""
        <div class="{cls_name}">
            <div style="margin-bottom:0.45rem;">
                {type_badge}
                <span class="riq-badge riq-badge-blue">{html.escape(gap.category)}</span>
                {diff_badge}
                <span class="riq-badge riq-badge-violet">Feasibility {gap.feasibility_score}/10 | Impact {gap.impact_score}/10</span>
            </div>
            <div style="font-size:1.02rem;font-weight:600;color:#F8FAFC;margin-bottom:0.35rem;">{html.escape(gap.title)}</div>
            <div style="font-size:0.88rem;color:#CBD5E1;margin-bottom:0.45rem;">{html.escape(gap.description)}</div>
            <div style="font-size:0.82rem;color:#94A3B8;">
                <b>Source:</b> {html.escape(gap.supporting_publication)} &nbsp;|&nbsp;
                <b>Page Ref:</b> {html.escape(gap.page_reference)}<br/>
                <b>Technical Significance:</b> {html.escape(gap.technical_significance or 'N/A')}<br/>
                <b>Potential Improvement:</b> {html.escape(gap.potential_improvement or 'N/A')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scholarly_work_card(work: ScholarlyWork) -> None:
    """Display scholarly publication metadata card without fabricating missing fields."""
    authors_str = ", ".join(work.authors[:4]) + (" et al." if len(work.authors) > 4 else "")
    cite_txt = f"Citations: {work.citation_count}" if work.citation_count is not None else "Citations: N/A"
    doi_txt = f"DOI: {work.doi}" if work.doi else "DOI: N/A"
    link_html = f'<a href="{html.escape(work.url)}" target="_blank" style="color:#A78BFA;text-decoration:none;">Open Publication ↗</a>' if work.url else ""
    st.markdown(
        f"""
        <div class="riq-card">
            <div style="margin-bottom:0.3rem;">
                <span class="riq-badge riq-badge-violet">{html.escape(work.source.upper())}</span>
                <span class="riq-badge riq-badge-blue">{html.escape(str(work.year or 'Year N/A'))}</span>
                <span class="riq-badge riq-badge-amber">{html.escape(cite_txt)}</span>
                <span style="float:right;font-size:0.8rem;">{link_html}</span>
            </div>
            <div style="font-weight:600;color:#F3F5FA;font-size:0.98rem;margin-bottom:0.25rem;">{html.escape(work.title)}</div>
            <div style="font-size:0.8rem;color:#94A3B8;margin-bottom:0.35rem;">
                {html.escape(authors_str or 'Authors not listed')} &bull; {html.escape(work.venue or 'Venue N/A')} &bull; {html.escape(doi_txt)}
            </div>
            <div style="font-size:0.82rem;color:#CBD5E1;">{html.escape((work.abstract or 'Abstract not provided by source API.')[:260])}...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
