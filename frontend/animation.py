"""High-contrast Soft-Slate & Violet SaaS design system, CSS animations, and live pipeline tracker."""

from __future__ import annotations

import html
import streamlit as st


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #F4F6FB;
    --bg-surface: #FFFFFF;
    --bg-surface-2: #EEF2FF;
    --bg-glass: rgba(255, 255, 255, 0.92);
    --border-subtle: #CBD5E1;
    --border-accent: #7C3AED;
    --text-primary: #0F172A;
    --text-secondary: #1E293B;
    --text-muted: #475569;
    --accent-violet: #6D28D9;
    --accent-purple: #5B21B6;
    --success: #059669;
    --warning: #D97706;
    --danger: #E11D48;
    --info: #0284C7;
}

/* Base layout & typography */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 10% 0%, rgba(124, 58, 237, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 92% 8%, rgba(14, 165, 233, 0.07) 0%, transparent 38%),
                var(--bg-primary);
}

.stMarkdown, .stMarkdown p, .stMarkdown li, label, .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-primary) !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling & smooth transitions */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    transition: width 0.28s ease, background-color 0.28s ease;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

/* Keyframe animations */
@keyframes riqFadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes riqGradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes riqPulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(109, 40, 217, 0.24); }
    50% { box-shadow: 0 0 16px 2px rgba(109, 40, 217, 0.18); }
}

/* Hero & Cards */
.riq-hero {
    background: linear-gradient(135deg, #EDE9FE 0%, #FFFFFF 55%, #E0F2FE 100%);
    background-size: 200% 200%;
    animation: riqFadeSlideUp 0.45s ease-out, riqGradientShift 14s ease infinite;
    border: 1px solid #C4B5FD;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.riq-card {
    background: var(--bg-surface);
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: riqFadeSlideUp 0.35s ease-out;
}

.riq-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-violet);
    box-shadow: 0 8px 20px rgba(109, 40, 217, 0.10);
}

.riq-glass {
    background: var(--bg-glass);
    backdrop-filter: blur(8px);
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

/* Metric cards */
.riq-metric {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease, border-color 0.2s ease;
    animation: riqFadeSlideUp 0.4s ease-out;
}
.riq-metric:hover {
    transform: translateY(-2px);
    border-color: var(--accent-violet);
}
.riq-metric-label {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.riq-metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.15;
}
.riq-metric-sub {
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--accent-violet);
    margin-top: 0.25rem;
}

/* Gap card variants (explicit vs inferred) */
.riq-gap-explicit {
    background: linear-gradient(135deg, #ECFDF5 0%, #FFFFFF 65%);
    border: 1px solid #6EE7B7;
    border-left: 5px solid #059669;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease;
}
.riq-gap-inferred {
    background: linear-gradient(135deg, #F5F3FF 0%, #FFFFFF 65%);
    border: 1px dashed #A78BFA;
    border-left: 5px solid #6D28D9;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s ease;
}
.riq-gap-explicit:hover, .riq-gap-inferred:hover {
    transform: translateY(-2px);
}

/* Badges */
.riq-badge {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 0.35rem;
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
}
.riq-badge-violet { background: #EDE9FE; color: #5B21B6; border: 1px solid #C4B5FD; }
.riq-badge-green { background: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }
.riq-badge-amber { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.riq-badge-red { background: #FFE4E6; color: #9F1239; border: 1px solid #FDA4AF; }
.riq-badge-blue { background: #E0F2FE; color: #075985; border: 1px solid #7DD3FC; }

/* Animated workflow ribbon */
.riq-workflow-step {
    padding: 0.55rem 0.65rem;
    border-radius: 10px;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    text-align: center;
    transition: all 0.22s ease;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}
.riq-workflow-step.completed {
    border-color: #34D399;
    background: #ECFDF5;
}
.riq-workflow-step.active {
    border-color: #7C3AED;
    background: #F5F3FF;
    animation: riqPulseGlow 2.4s infinite;
}

/* Interactive buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 9px !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #6D28D9 !important;
    background: #F5F3FF !important;
    color: #5B21B6 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(109, 40, 217, 0.14);
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED, #5B21B6) !important;
    border-color: #5B21B6 !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6D28D9, #4C1D95) !important;
    color: #FFFFFF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #E2E8F0;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #334155 !important;
    font-weight: 600;
    padding: 8px 14px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #5B21B6 !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
}

/* Inputs, Expanders & Containers */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 12px;
}

/* Reduced motion accessibility */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


def inject_theme_css() -> None:
    """Inject RESEARCHIQ high-contrast Soft-Slate & Violet CSS design system."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_real_processing_tracker(
    completed_stages: list[str],
    active_stage: str | None = None,
) -> None:
    """
    Render real backend processing stages without fake percentages.
    """
    stages = [
        "Document uploaded",
        "Text extracted",
        "Sections detected",
        "Embeddings generated",
        "Research retrieved",
        "Analysis performed",
        "Results generated",
    ]
    pills = []
    for idx, stg in enumerate(stages, start=1):
        if stg in completed_stages:
            cls = "riq-badge riq-badge-green"
            icon = "✓"
        elif active_stage == stg:
            cls = "riq-badge riq-badge-violet"
            icon = "●"
        else:
            cls = "riq-badge"
            icon = f"{idx}"
        pills.append(f'<span class="{cls}">{icon} {html.escape(stg)}</span>')
    st.markdown(
        f'<div class="riq-glass" style="margin-bottom:0.75rem;">'
        f'<div style="font-size:0.75rem;color:#475569;font-weight:700;margin-bottom:0.35rem;">LIVE PROCESSING PIPELINE</div>'
        f'{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )
