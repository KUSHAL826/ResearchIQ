"""Interactive Plotly, Mermaid.js, PRISMA, and hierarchy visualization components."""

from __future__ import annotations

import html
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core.schemas import (
    PRISMACounts,
    PaperAnalysis,
    ParsedPaper,
    ResearchGap,
    ScholarlyWork,
)

PLOTLY_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(19, 23, 34, 0.0)",
    plot_bgcolor="rgba(19, 23, 34, 0.55)",
    font=dict(family="Inter, sans-serif", color="#E2E8F0", size=12),
    margin=dict(l=35, r=25, t=45, b=35),
)


def render_mermaid_diagram(mermaid_code: str, height: int = 360, title: str = "") -> None:
    """Render a Mermaid.js diagram inside a dark glass panel with live SVG rendering."""
    if not mermaid_code or not mermaid_code.strip():
        st.info("No diagram specification available yet.")
        return
    uid = f"mermaid_{uuid4().hex[:8]}"
    safe_code = html.escape(mermaid_code.strip())
    header = f'<div style="font-size:0.8rem;color:#A78BFA;font-weight:600;margin-bottom:6px;">{html.escape(title)}</div>' if title else ""
    html_doc = f"""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 12px;
          background: #111521;
          color: #F3F5FA;
          font-family: Inter, sans-serif;
          border-radius: 12px;
        }}
        .mermaid {{
          display: flex;
          justify-content: center;
          overflow: auto;
        }}
      </style>
    </head>
    <body>
      {header}
      <div id="{uid}" class="mermaid">{safe_code}</div>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'dark',
          themeVariables: {{
            primaryColor: '#2B2250',
            primaryTextColor: '#F3F5FA',
            primaryBorderColor: '#8B5CF6',
            lineColor: '#A78BFA',
            secondaryColor: '#192236',
            tertiaryColor: '#131722'
          }}
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_doc, height=height, scrolling=True)


def fig_paper_structure_tree(paper: ParsedPaper) -> go.Figure:
    """Interactive Treemap showing parsed paper sections and page/character coverage."""
    if not paper.sections:
        return _empty_figure("No sections parsed yet")
    labels = [paper.title[:35]]
    parents = [""]
    values = [sum(max(100, len(s.content)) for s in paper.sections)]
    texts = [f"{paper.num_pages} pages"]

    for sec in paper.sections:
        labels.append(f"{sec.canonical_name} (pp.{sec.start_page}-{sec.end_page})")
        parents.append(paper.title[:35])
        values.append(max(100, len(sec.content)))
        texts.append(f"Heading: {sec.raw_heading}<br>Pages: {sec.start_page}-{sec.end_page}<br>Chars: {len(sec.content)}")

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            text=texts,
            hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
            marker=dict(
                colorscale=[[0, "#1E1B4B"], [0.5, "#6D28D9"], [1, "#A78BFA"]],
                line=dict(color="#0B0D12", width=2),
            ),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, title="Interactive Paper Structure Map", height=320)
    return fig


def fig_paper_results_chart(analysis: PaperAnalysis) -> go.Figure:
    """Plot numeric metrics extracted from an analyzed paper."""
    numeric = [r for r in analysis.results if r.value is not None]
    if not numeric:
        return _empty_figure("No numeric metrics extracted from this paper")
    df = pd.DataFrame(
        [
            {
                "Metric": f"{r.metric} ({r.page_reference})",
                "Value": r.value,
                "Model": r.model_or_method or "Reported",
            }
            for r in numeric
        ]
    )
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Model",
        title="Extracted Quantitative Metrics",
        color_discrete_sequence=["#8B5CF6", "#38BDF8", "#10B981", "#F59E0B"],
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
    return fig


def fig_gap_feasibility_impact(gaps: list[ResearchGap]) -> go.Figure:
    """Interactive Feasibility vs Impact scatter plot distinguishing Explicit vs Inferred gaps."""
    if not gaps:
        return _empty_figure("Run Gap Detective to populate the Feasibility vs Impact map")
    df = pd.DataFrame(
        [
            {
                "Title": g.title,
                "Feasibility": g.feasibility_score,
                "Impact": g.impact_score,
                "Classification": "Author-Stated (Explicit)" if g.classification == "explicit" else "AI-Inferred",
                "Category": g.category,
                "Source": f"{g.supporting_publication[:35]} ({g.page_reference})",
            }
            for g in gaps
        ]
    )
    fig = px.scatter(
        df,
        x="Feasibility",
        y="Impact",
        color="Classification",
        symbol="Category",
        hover_name="Title",
        hover_data=["Source", "Category"],
        title="Research Opportunity Map (Feasibility vs Impact)",
        color_discrete_map={"Author-Stated (Explicit)": "#10B981", "AI-Inferred": "#8B5CF6"},
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color="#0B0D12")))
    fig.update_xaxes(range=[0.5, 10.5], title="Feasibility Score (1-10)")
    fig.update_yaxes(range=[0.5, 10.5], title="Research Impact Score (1-10)")
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=380)
    return fig


def fig_gap_classification_counts(gaps: list[ResearchGap]) -> go.Figure:
    """Bar breakdown of gaps by category and classification."""
    if not gaps:
        return _empty_figure("No research gaps identified yet")
    df = pd.DataFrame(
        [
            {
                "Category": g.category,
                "Classification": "Explicit" if g.classification == "explicit" else "Inferred",
            }
            for g in gaps
        ]
    )
    grouped = df.groupby(["Category", "Classification"]).size().reset_index(name="Count")
    fig = px.bar(
        grouped,
        x="Category",
        y="Count",
        color="Classification",
        barmode="group",
        title="Gap Classification & Category Breakdown",
        color_discrete_map={"Explicit": "#10B981", "Inferred": "#8B5CF6"},
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
    return fig


def fig_prisma_flow(prisma: PRISMACounts) -> go.Figure:
    """Interactive PRISMA Sankey Flow Diagram driven strictly by real screening counts."""
    if prisma.records_identified == 0:
        return _empty_figure("Discover or add papers to generate your live PRISMA workflow")

    labels = [
        f"Records Identified ({prisma.records_identified})",
        f"Duplicates Removed ({prisma.duplicates_removed})",
        f"Records Screened ({prisma.records_screened})",
        f"Excluded at Screening ({prisma.records_excluded})",
        f"Full-Text Assessed ({prisma.full_text_assessed})",
        f"Excluded Full-Text ({prisma.full_text_excluded})",
        f"Studies Included ({prisma.studies_included})",
        f"Needs Review ({prisma.needs_review})",
    ]
    sources = [0, 0, 2, 2, 2, 4, 4]
    targets = [1, 2, 3, 4, 7, 5, 6]
    values = [
        max(0, prisma.duplicates_removed),
        max(0, prisma.records_screened),
        max(0, prisma.records_excluded),
        max(0, prisma.full_text_assessed),
        max(0, prisma.needs_review),
        max(0, prisma.full_text_excluded),
        max(0, prisma.studies_included),
    ]
    # Keep only non-zero flows for clean Sankey
    s_nz, t_nz, v_nz = [], [], []
    for s, t, v in zip(sources, targets, values):
        if v > 0:
            s_nz.append(s)
            t_nz.append(t)
            v_nz.append(v)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=18,
                thickness=22,
                line=dict(color="#1E293B", width=1),
                label=labels,
                color=["#7C3AED", "#64748B", "#38BDF8", "#F43F5E", "#8B5CF6", "#FB7185", "#10B981", "#F59E0B"],
            ),
            link=dict(
                source=s_nz,
                target=t_nz,
                value=v_nz,
                color="rgba(139, 92, 246, 0.28)",
            ),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, title="Interactive PRISMA Screening Flow", height=360)
    return fig


def fig_publication_timeline(works: list[ScholarlyWork]) -> go.Figure:
    """Yearly publication count chart from real retrieved works."""
    unique = [w for w in works if not w.is_duplicate and w.year]
    if not unique:
        return _empty_figure("No dated publications available yet")
    df = pd.DataFrame([{"Year": w.year, "Source": w.source.upper()} for w in unique])
    counts = df.groupby(["Year", "Source"]).size().reset_index(name="Publications")
    fig = px.bar(
        counts,
        x="Year",
        y="Publications",
        color="Source",
        title="Publication Year Distribution by Source",
        color_discrete_sequence=["#8B5CF6", "#38BDF8", "#10B981", "#F59E0B"],
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=340)
    return fig


def fig_trends_heatmap(topic_year_matrix: dict[str, dict[int, int]]) -> go.Figure:
    """Topic vs Year publication heatmap from real metadata."""
    if not topic_year_matrix:
        return _empty_figure("Fetch scholarly works to render the Topic-Year Heatmap")
    df = pd.DataFrame(topic_year_matrix).T.fillna(0)
    df = df.reindex(sorted(df.columns), axis=1)
    fig = px.imshow(
        df,
        labels=dict(x="Publication Year", y="Research Topic", color="Papers"),
        title="Topic vs Publication Year Heatmap",
        color_continuous_scale="Purples",
        aspect="auto",
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=360)
    return fig


def fig_opportunity_radar(opportunity_rows: list[dict[str, Any]]) -> go.Figure:
    """Polar radar chart of topic momentum vs gap opportunity."""
    if not opportunity_rows:
        return _empty_figure("No topic opportunity scores available yet")
    top = opportunity_rows[:6]
    topics = [r["Topic"] for r in top]
    momentum = [r["Recent Momentum"] for r in top]
    gap_opp = [r["Gap Opportunity"] for r in top]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=momentum, theta=topics, fill="toself", name="Recent Momentum", line_color="#38BDF8"))
    fig.add_trace(go.Scatterpolar(r=gap_opp, theta=topics, fill="toself", name="Gap Opportunity", line_color="#8B5CF6"))
    fig.update_layout(
        **PLOTLY_LAYOUT_BASE,
        polar=dict(bgcolor="rgba(19, 23, 34, 0.7)", radialaxis=dict(visible=True, range=[0, 10])),
        title="Research Opportunity Radar",
        height=380,
    )
    return fig


def fig_experiment_comparison(df: pd.DataFrame, x_col: str, y_cols: list[str]) -> go.Figure:
    """Bar chart comparing actual uploaded experimental results across models/configs."""
    if df.empty or not y_cols or x_col not in df.columns:
        return _empty_figure("Select categorical/model column and numeric metric columns")
    fig = go.Figure()
    palette = ["#8B5CF6", "#38BDF8", "#10B981", "#F59E0B", "#EC4899"]
    for idx, yc in enumerate(y_cols):
        if yc in df.columns:
            fig.add_trace(
                go.Bar(
                    name=yc,
                    x=df[x_col].astype(str),
                    y=pd.to_numeric(df[yc], errors="coerce"),
                    marker_color=palette[idx % len(palette)],
                )
            )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, barmode="group", title="Experimental Performance Comparison", height=370)
    return fig


def fig_confusion_matrix(cm: list[list[int]], labels: list[str]) -> go.Figure:
    """Interactive confusion matrix heatmap computed from actual uploaded prediction columns."""
    if not cm or not labels:
        return _empty_figure("No confusion matrix computed yet")
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        title="Confusion Matrix (Actual Uploaded Predictions)",
        color_continuous_scale="Purples",
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, height=360)
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#94A3B8"),
    )
    fig.update_layout(**PLOTLY_LAYOUT_BASE, xaxis=dict(visible=False), yaxis=dict(visible=False), height=280)
    return fig
