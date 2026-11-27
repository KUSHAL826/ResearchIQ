"""Part 2.1: Unified Project Research Knowledge Graph (Source Evidence vs AI-Inferred)."""

from __future__ import annotations

from collections import Counter
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network

from core.schemas import ResearchProject


NODE_PALETTE = {
    "problem": "#DC2626",
    "objective": "#EA580C",
    "question": "#D97706",
    "paper": "#6D28D9",
    "author": "#0284C7",
    "method": "#7C3AED",
    "algorithm": "#4F46E5",
    "dataset": "#0D9488",
    "metric": "#059669",
    "gap": "#E11D48",
    "limitation": "#F43F5E",
    "finding": "#16A34A",
    "citation": "#64748B",
}


def build_unified_knowledge_graph(project: ResearchProject) -> nx.DiGraph:
    """
    Build a comprehensive project-level knowledge graph connecting:
    papers, authors, problems, objectives, methods, algorithms, datasets,
    metrics, gaps, findings, limitations, citations, and research questions.
    Every edge specifies evidence_type ('Source Evidence' or 'AI-Inferred') and source_ref.
    """
    G = nx.DiGraph()

    def _add(nid: str, label: str, ntype: str, detail: str = "") -> None:
        if nid not in G:
            G.add_node(nid, label=label[:36], full_title=label, node_type=ntype, detail=detail)

    # Problem, Objectives & Research Questions
    prob_txt = project.problem_statement.problem_definition or project.selected_topic or project.title
    prob_id = "problem::root"
    _add(prob_id, f"Problem: {prob_txt[:60]}", "problem", prob_txt)

    for idx, obj in enumerate(project.problem_statement.objectives, start=1):
        oid = f"obj::{idx}"
        _add(oid, f"O{idx}: {obj[:50]}", "objective", obj)
        G.add_edge(prob_id, oid, relation="decomposes_into", evidence_type="Source Evidence", source_ref="Stage 2 Objectives")

    for idx, rq in enumerate(project.problem_statement.research_questions, start=1):
        rqid = f"rq::{idx}"
        _add(rqid, f"RQ{idx}: {rq[:50]}", "question", rq)
        G.add_edge(prob_id, rqid, relation="investigates", evidence_type="Source Evidence", source_ref="Stage 2 Research Questions")

    # Uploaded PDFs & Deep Paper Intelligence
    for p in project.parsed_papers:
        pid = f"paper::{p.paper_id}"
        _add(pid, p.title, "paper", f"Uploaded PDF ({p.num_pages} pp., year={p.year or 'N/A'})")
        G.add_edge(prob_id, pid, relation="informed_by", evidence_type="Source Evidence", source_ref=p.filename)

        for auth in p.authors[:4]:
            aid = f"author::{auth}"
            _add(aid, auth, "author")
            G.add_edge(pid, aid, relation="authored_by", evidence_type="Source Evidence", source_ref=p.filename)

        analysis = project.paper_analyses.get(p.paper_id)
        if analysis:
            for step in analysis.methodology_steps[:4]:
                mid = f"method::{step[:32]}"
                _add(mid, step, "method", step)
                G.add_edge(pid, mid, relation="uses_method", evidence_type="Source Evidence", source_ref=p.title)

            for alg in analysis.algorithms[:5]:
                alid = f"alg::{alg.name}"
                _add(alid, alg.name, "algorithm", f"{alg.role} ({alg.page_reference})")
                G.add_edge(pid, alid, relation="applies_algorithm", evidence_type="Source Evidence", source_ref=f"{p.title} ({alg.page_reference})")

            for ds in analysis.datasets[:4]:
                dsid = f"dataset::{ds.name}"
                _add(dsid, ds.name, "dataset", f"{ds.domain} ({ds.page_reference})")
                G.add_edge(pid, dsid, relation="evaluates_on", evidence_type="Source Evidence", source_ref=f"{p.title} ({ds.page_reference})")

            for res in analysis.results[:4]:
                met_id = f"metric::{res.metric}"
                _add(met_id, res.metric, "metric")
                find_id = f"finding::{p.paper_id}::{res.metric}"
                _add(find_id, f"{res.metric}={res.raw_value or res.value}", "finding", f"{p.title} ({res.page_reference})")
                G.add_edge(pid, find_id, relation="reports_finding", evidence_type="Source Evidence", source_ref=res.page_reference)
                G.add_edge(find_id, met_id, relation="measured_by", evidence_type="Source Evidence", source_ref=res.page_reference)

            for lim in analysis.limitations[:4]:
                lid = f"lim::{p.paper_id}::{lim.description[:25]}"
                _add(lid, lim.description[:50], "limitation", f"{lim.description} ({lim.page_reference})")
                G.add_edge(pid, lid, relation="has_limitation", evidence_type="Source Evidence", source_ref=lim.page_reference)

    # Discovered & Screened Works
    for w in [wk for wk in project.discovered_works if not wk.is_duplicate][:18]:
        wid = f"work::{w.work_id}"
        _add(wid, w.title, "citation" if w.screening_status != "included" else "paper", f"{w.source} ({w.year or 'n.d.'}) [{w.screening_status}]")
        G.add_edge(prob_id, wid, relation="literature_context", evidence_type="Source Evidence", source_ref=w.source)
        for auth in w.authors[:2]:
            aid = f"author::{auth}"
            _add(aid, auth, "author")
            G.add_edge(wid, aid, relation="authored_by", evidence_type="Source Evidence", source_ref=w.title)

    # Research Gaps -> Connect Source Paper / Limitation -> Proposed Direction
    for g in project.research_gaps:
        gid = f"gap::{g.gap_id}"
        ev_type = "Source Evidence" if g.classification == "explicit" else "AI-Inferred"
        _add(gid, g.title, "gap", f"{g.description} ({g.supporting_publication}, {g.page_reference})")
        if g.paper_id and f"paper::{g.paper_id}" in G:
            G.add_edge(f"paper::{g.paper_id}", gid, relation="reveals_gap", evidence_type=ev_type, source_ref=f"{g.supporting_publication} ({g.page_reference})")
        G.add_edge(gid, prob_id, relation="motivates_research", evidence_type=ev_type, source_ref=g.page_reference)

    # Proposed Methodology & Algorithms (Studio Stage 6)
    for alg in project.methodology.algorithms:
        aname = alg.get("name", "Proposed Module")
        alid = f"alg::{aname}"
        _add(alid, aname, "algorithm", alg.get("role", ""))
        G.add_edge(prob_id, alid, relation="proposes_algorithm", evidence_type="AI-Inferred", source_ref="Stage 6 Methodology")

    return G


def summarize_unified_graph(G: nx.DiGraph) -> dict[str, Any]:
    """Summarize node counts, edge evidence types, and top central entities."""
    node_types = Counter(d.get("node_type", "other") for _, d in G.nodes(data=True))
    edge_evidence = Counter(d.get("evidence_type", "Source Evidence") for _, _, d in G.edges(data=True))
    undirected = G.to_undirected()
    top_hubs = sorted(
        [
            {
                "Entity": G.nodes[n].get("full_title", n)[:60],
                "Type": G.nodes[n].get("node_type", ""),
                "Connections": undirected.degree(n),
            }
            for n in G.nodes()
        ],
        key=lambda x: x["Connections"],
        reverse=True,
    )[:12]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": dict(node_types),
        "edge_evidence_types": dict(edge_evidence),
        "top_hubs": top_hubs,
    }


def render_unified_kg_pyvis(
    G: nx.DiGraph,
    filter_types: list[str] | None = None,
    search_query: str = "",
    height_px: int = 540,
) -> str:
    """Render filtered Unified Knowledge Graph via PyVis on a crisp light canvas."""
    sub_nodes = [
        n for n, d in G.nodes(data=True)
        if not filter_types or d.get("node_type") in filter_types
    ]
    sub = G.subgraph(sub_nodes)
    net = Network(height=f"{height_px}px", width="100%", bgcolor="#F8FAFC", font_color="#0F172A", directed=True)
    net.barnes_hut(gravity=-3500, central_gravity=0.32, spring_length=140, spring_strength=0.04)

    q = search_query.strip().lower()
    for nid, data in sub.nodes(data=True):
        ntype = data.get("node_type", "paper")
        full = str(data.get("full_title", nid))
        is_hit = bool(q and q in full.lower())
        color = "#F59E0B" if is_hit else NODE_PALETTE.get(ntype, "#6D28D9")
        tooltip = f"<b>{full}</b><br>Type: {ntype}<br>{data.get('detail', '')}"
        net.add_node(
            nid,
            label=data.get("label", nid[:28]),
            title=tooltip,
            color=color,
            size=26 if ntype in ("problem", "paper", "gap") else 16,
            borderWidth=3 if is_hit else 1,
        )

    for src, dst, edata in sub.edges(data=True):
        ev_type = edata.get("evidence_type", "Source Evidence")
        rel = edata.get("relation", "")
        net.add_edge(
            src,
            dst,
            title=f"{rel} [{ev_type}] — {edata.get('source_ref', '')}",
            color="#059669" if ev_type == "Source Evidence" else "#7C3AED",
            dashes=(ev_type == "AI-Inferred"),
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.save_graph(str(tmp_path))
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)
