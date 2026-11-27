"""Feature 6: Citation Network Explorer using NetworkX analytics and interactive PyVis rendering."""

from __future__ import annotations

from collections import Counter
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

from core.schemas import PaperAnalysis, ParsedPaper, ScholarlyWork


def build_citation_graph(
    works: list[ScholarlyWork],
    parsed_papers: list[ParsedPaper] | None = None,
    analyses: dict[str, PaperAnalysis] | None = None,
) -> nx.DiGraph:
    """
    Build a multi-type research graph with:
    Nodes: research papers, authors, research topics, cited publications.
    Edges: cites, written_by, related_to, shares_methodology.
    """
    G = nx.DiGraph()
    analyses = analyses or {}
    parsed_papers = parsed_papers or []

    # Add uploaded papers
    for p in parsed_papers:
        G.add_node(
            p.paper_id,
            label=p.title[:42] + ("..." if len(p.title) > 42 else ""),
            full_title=p.title,
            node_type="uploaded_paper",
            year=p.year or "N/A",
            citations="Uploaded PDF",
            doi=p.doi or "N/A",
            summary=p.hierarchical_summary[:280],
        )
        for author in p.authors[:5]:
            aid = f"author::{author}"
            if aid not in G:
                G.add_node(aid, label=author, full_title=author, node_type="author")
            G.add_edge(p.paper_id, aid, relation="written_by")

        analysis = analyses.get(p.paper_id)
        if analysis:
            for alg in analysis.algorithms[:4]:
                tid = f"topic::{alg.name}"
                if tid not in G:
                    G.add_node(tid, label=alg.name, full_title=alg.name, node_type="topic")
                G.add_edge(p.paper_id, tid, relation="shares_methodology")

    # Add scholarly works
    unique_works = [w for w in works if not w.is_duplicate]
    work_ids = {w.work_id for w in unique_works}

    for w in unique_works:
        G.add_node(
            w.work_id,
            label=w.title[:40] + ("..." if len(w.title) > 40 else ""),
            full_title=w.title,
            node_type="paper",
            year=w.year or "N/A",
            citations=w.citation_count if w.citation_count is not None else "Unknown",
            doi=w.doi or "N/A",
            venue=w.venue or w.source,
            url=w.url,
            summary=(w.abstract or "")[:280],
        )
        for author in w.authors[:3]:
            aid = f"author::{author}"
            if aid not in G:
                G.add_node(aid, label=author, full_title=author, node_type="author")
            G.add_edge(w.work_id, aid, relation="written_by")

        for field in w.fields_of_study[:3]:
            tid = f"topic::{field}"
            if tid not in G:
                G.add_node(tid, label=field, full_title=field, node_type="topic")
            G.add_edge(w.work_id, tid, relation="related_to")

        # Citation edges
        for ref_id in w.referenced_ids[:6]:
            if ref_id not in G and ref_id not in work_ids:
                G.add_node(
                    ref_id,
                    label=f"Ref {ref_id[:14]}",
                    full_title=f"Referenced Work ({ref_id})",
                    node_type="cited_publication",
                )
            G.add_edge(w.work_id, ref_id, relation="cites")

    # Connect papers sharing methodology keywords in title/abstract
    method_keywords = ["transformer", "graph neural", "rag", "retrieval", "diffusion", "reinforcement learning", "bert", "lstm", "cnn"]
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") in ("paper", "uploaded_paper")]
    for kw in method_keywords:
        matching = [
            n
            for n in paper_nodes
            if kw in f"{G.nodes[n].get('full_title', '')} {G.nodes[n].get('summary', '')}".lower()
        ]
        for i in range(len(matching)):
            for j in range(i + 1, min(len(matching), i + 4)):
                G.add_edge(matching[i], matching[j], relation="shares_methodology")

    return G


def analyze_graph_metrics(G: nx.DiGraph) -> dict[str, Any]:
    """Compute centrality and influential publication rankings using NetworkX."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "central_papers": [], "node_type_counts": {}}

    undirected = G.to_undirected()
    deg = nx.degree_centrality(undirected)
    try:
        pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}
    except Exception:
        pr = deg

    type_counts = Counter(d.get("node_type", "other") for _, d in G.nodes(data=True))

    central_papers = []
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") in ("paper", "uploaded_paper"):
            central_papers.append(
                {
                    "id": node_id,
                    "title": data.get("full_title", node_id),
                    "year": data.get("year", "N/A"),
                    "citations": data.get("citations", "Unknown"),
                    "degree_centrality": round(deg.get(node_id, 0.0), 4),
                    "pagerank": round(pr.get(node_id, 0.0), 4),
                    "connections": undirected.degree(node_id),
                }
            )
    central_papers.sort(key=lambda x: (x["pagerank"], x["connections"]), reverse=True)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "central_papers": central_papers[:15],
        "node_type_counts": dict(type_counts),
    }


def render_pyvis_html(G: nx.DiGraph, search_highlight: str = "", height_px: int = 560) -> str:
    """Render NetworkX graph to dark-themed interactive PyVis HTML string."""
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#0D1017",
        font_color="#E5E7EB",
        directed=True,
    )
    net.barnes_hut(gravity=-3200, central_gravity=0.35, spring_length=135, spring_strength=0.04)

    palette = {
        "uploaded_paper": "#A78BFA",
        "paper": "#7C3AED",
        "author": "#38BDF8",
        "topic": "#10B981",
        "cited_publication": "#F59E0B",
    }
    edge_colors = {
        "cites": "#F59E0B",
        "written_by": "#38BDF8",
        "related_to": "#10B981",
        "shares_methodology": "#C084FC",
    }

    query = search_highlight.strip().lower()
    undirected = G.to_undirected()

    for node_id, data in G.nodes(data=True):
        ntype = data.get("node_type", "paper")
        title_full = str(data.get("full_title", node_id))
        is_match = bool(query and query in title_full.lower())
        color = "#F43F5E" if is_match else palette.get(ntype, "#8B5CF6")
        base_size = 24 if ntype in ("uploaded_paper", "paper") else 14
        size = base_size + min(18, undirected.degree(node_id) * 2) + (8 if is_match else 0)
        tooltip = (
            f"<b>{title_full}</b><br>"
            f"Type: {ntype.replace('_', ' ').title()}<br>"
            f"Year: {data.get('year', 'N/A')} | Citations: {data.get('citations', 'N/A')}<br>"
            f"Connections: {undirected.degree(node_id)}"
        )
        net.add_node(
            node_id,
            label=data.get("label", str(node_id)[:30]),
            title=tooltip,
            color=color,
            size=size,
            borderWidth=3 if is_match else 1,
        )

    for src, dst, edata in G.edges(data=True):
        rel = edata.get("relation", "related_to")
        net.add_edge(
            src,
            dst,
            title=rel.replace("_", " "),
            color=edge_colors.get(rel, "#64748B"),
            arrows="to" if rel in ("cites", "written_by") else "",
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.save_graph(str(tmp_path))
        html = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
    return html
