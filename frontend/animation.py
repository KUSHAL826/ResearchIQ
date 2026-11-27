"""Dark charcoal & violet SaaS design system, CSS keyframe animations, and real-stage progress UI."""

from __future__ import annotations

import html
import streamlit as st


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0B0D12;
    --bg-surface: #131722;
    --bg-surface-2: #191E2D;
    --bg-glass: rgba(25, 30, 45, 0.72);
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-accent: rgba(139, 92, 246, 0.45);
    --text-primary: #F3F5FA;
    --text-muted: #94A3B8;
    --accent-violet: #8B5CF6;
    --accent-purple: #7C3AED;
    --accent-glow: rgba(139, 92, 246, 0.22);
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #F43F5E;
    --info: #38BDF8;
}

/* Base layout & typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 12% 0%, rgba(124, 58, 237, 0.12) 0%, transparent 42%),
                radial-gradient(circle at 88% 12%, rgba(56, 189, 248, 0.06) 0%, transparent 38%),
                var(--bg-primary);
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling & smooth transitions */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E111A 0%, #111521 100%);
    border-right: 1px solid var(--border-subtle);
    transition: width 0.28s ease, background-color 0.28s ease;
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
    0%, 100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.35); }
    50% { box-shadow: 0 0 18px 2px rgba(139, 92, 246, 0.25); }
}

@keyframes riqConnectorFlow {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

/* Hero & Cards */
.riq-hero {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(19, 23, 34, 0.92) 55%, rgba(56, 189, 248, 0.08));
    background-size: 200% 200%;
    animation: riqFadeSlideUp 0.45s ease-out, riqGradientShift 14s ease infinite;
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
}

.riq-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    animation: riqFadeSlideUp 0.35s ease-out;
}

.riq-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-accent);
    box-shadow: 0 10px 26px rgba(124, 58, 237, 0.14);
}

.riq-glass {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
}

/* Metric cards */
.riq-metric {
    background: linear-gradient(180deg, #161B29 0%, #121622 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
    animation: riqFadeSlideUp 0.4s ease-out;
}
.riq-metric:hover {
    transform: translateY(-2px);
    border-color: var(--accent-violet);
}
.riq-metric-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.riq-metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
}
.riq-metric-sub {
    font-size: 0.75rem;
    color: #A78BFA;
    margin-top: 0.25rem;
}

/* Gap card variants (explicit vs inferred) */
.riq-gap-explicit {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.09), #131722 60%);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-left: 4px solid #10B981;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    transition: transform 0.2s ease;
}
.riq-gap-inferred {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.11), #131722 60%);
    border: 1px dashed rgba(139, 92, 246, 0.45);
    border-left: 4px solid #8B5CF6;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
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
}
.riq-badge-violet { background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.4); }
.riq-badge-green { background: rgba(16, 185, 129, 0.18); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.4); }
.riq-badge-amber { background: rgba(245, 158, 11, 0.18); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
.riq-badge-red { background: rgba(244, 63, 94, 0.18); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); }
.riq-badge-blue { background: rgba(56, 189, 248, 0.18); color: #7DD3FC; border: 1px solid rgba(56, 189, 248, 0.4); }

/* Animated workflow ribbon */
.riq-workflow-step {
    padding: 0.55rem 0.65rem;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    background: var(--bg-surface);
    text-align: center;
    transition: all 0.22s ease;
}
.riq-workflow-step.completed {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.08);
}
.riq-workflow-step.active {
    border-color: var(--accent-violet);
    background: rgba(139, 92, 246, 0.16);
    animation: riqPulseGlow 2.4s infinite;
}

/* Interactive buttons */
.stButton > button {
    border-radius: 9px !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    background: linear-gradient(135deg, #1E2235, #171B2A) !important;
    color: #F3F5FA !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    border-color: #A78BFA !important;
    background: linear-gradient(135deg, #2B244D, #1E1B38) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.25);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    border-color: #A78BFA !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #11141E;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-muted);
    padding: 8px 14px;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: rgba(139, 92, 246, 0.2) !important;
    color: #F3F5FA !important;
    border: 1px solid rgba(139, 92, 246, 0.45);
}

/* Inputs & Expanders */
div[data-testid="stExpander"] {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
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
    """Inject RESEARCHIQ dark charcoal & violet CSS design system."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_real_processing_tracker(
    completed_stages: list[str],
    active_stage: str | None = None,
) -> None:
    """
    Render real backend processing stages without fake percentages.
    Standard stages:
    1. Document uploaded
    2. Text extracted
    3. Sections detected
    4. Embeddings generated
    5. Research retrieved
    6. Analysis performed
    7. Results generated
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
        f'<div style="font-size:0.75rem;color:#94A3B8;margin-bottom:0.35rem;">LIVE PROCESSING PIPELINE</div>'
        f'{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )
