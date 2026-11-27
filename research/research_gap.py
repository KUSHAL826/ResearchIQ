"""Feature 2: Research Gap Detective — explicit vs AI-inferred gaps with page-level evidence."""

from __future__ import annotations

import re
from typing import List

import networkx as nx
from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import PaperAnalysis, ParsedPaper, ResearchGap


class GapListPayload(BaseModel):
    gaps: List[ResearchGap] = Field(default_factory=list)


def detect_research_gaps(
    papers: list[ParsedPaper],
    analyses: dict[str, PaperAnalysis] | None = None,
    llm_client: GeminiClient | None = None,
) -> list[ResearchGap]:
    """Identify explicit (author-stated) and AI-inferred research gaps grounded in parsed papers."""
    if not papers:
        return []
    analyses = analyses or {}
    gaps: list[ResearchGap] = []

    if llm_client and llm_client.is_configured:
        for paper in papers[:6]:
            analysis = analyses.get(paper.paper_id)
            lim_summary = "\n".join(
                f"- ({l.page_reference}) {l.description}" for l in (analysis.limitations if analysis else [])
            )
            relevant_sections = "\n\n".join(
                f"[{s.canonical_name} pp.{s.start_page}-{s.end_page}]\n{s.content[:2000]}"
                for s in paper.sections
                if s.canonical_name in ("Abstract", "Introduction", "Literature Review", "Results", "Conclusion")
            ) or "\n".join(p.text[:1500] for p in paper.pages[:4])

            prompt = (
                f"Identify 3 to 5 concrete research gaps from this paper.\n"
                f"Paper Title: {paper.title}\nPaper ID: {paper.paper_id}\n"
                f"Extracted Limitations:\n{lim_summary or 'None pre-extracted'}\n\n"
                f"RULES:\n"
                f"- Set classification='explicit' ONLY if the authors explicitly state the limitation/gap, and quote evidence_quote + exact page_reference.\n"
                f"- Set classification='inferred' if synthesized across methodology/datasets/results.\n"
                f"- Provide realistic feasibility_score (1-10) and impact_score (1-10).\n\n"
                f"PAPER EXCERPTS:\n{relevant_sections[:12000]}"
            )
            try:
                payload = llm_client.generate_structured(prompt, GapListPayload)
                for g in payload.gaps:
                    g.supporting_publication = paper.title
                    g.paper_id = paper.paper_id
                    gaps.append(g)
                continue
            except Exception:
                pass

            gaps.extend(_heuristic_gaps_for_paper(paper, analysis))
        return gaps

    for paper in papers:
        gaps.extend(_heuristic_gaps_for_paper(paper, analyses.get(paper.paper_id)))
    return gaps


def _heuristic_gaps_for_paper(paper: ParsedPaper, analysis: PaperAnalysis | None) -> list[ResearchGap]:
    found: list[ResearchGap] = []
    # 1. Explicit gaps from extracted limitations
    if analysis and analysis.limitations:
        for idx, lim in enumerate(analysis.limitations[:4]):
            found.append(
                ResearchGap(
                    title=f"Author-Stated Limitation #{idx + 1}",
                    description=lim.description,
                    supporting_publication=paper.title,
                    paper_id=paper.paper_id,
                    page_reference=lim.page_reference or "N/A",
                    classification="explicit",
                    category="Methodological",
                    technical_significance="Directly reported constraint affecting generalisability or performance.",
                    potential_improvement="Address this constraint through targeted architectural or evaluation extensions.",
                    feasibility_score=7.0,
                    impact_score=7.5,
                    implementation_difficulty="Medium",
                    evidence_quote=lim.source_quote or lim.description[:200],
                )
            )
    else:
        # Scan pages for explicit limitation/future-work sentences
        pat = re.compile(r"([^.\n]*\b(?:future work|limitation|remain[s]? open|not addressed|drawback|scalability issue)\b[^.\n]*\.)", re.IGNORECASE)
        for p in paper.pages:
            for m in pat.finditer(p.text):
                sent = re.sub(r"\s+", " ", m.group(1)).strip()
                if len(sent) > 25:
                    found.append(
                        ResearchGap(
                            title=f"Explicit Gap (p. {p.page_number})",
                            description=sent,
                            supporting_publication=paper.title,
                            paper_id=paper.paper_id,
                            page_reference=f"p. {p.page_number}",
                            classification="explicit",
                            category="Methodological",
                            technical_significance="Identified directly in source text.",
                            potential_improvement="Design targeted experiment or extension resolving this bottleneck.",
                            feasibility_score=7.0,
                            impact_score=7.5,
                            implementation_difficulty="Medium",
                            evidence_quote=sent[:220],
                        )
                    )
                if len(found) >= 3:
                    break
            if len(found) >= 3:
                break

    # 2. Inferred gap checks from structural evidence (e.g. single dataset / missing runtime / ablation)
    full_text = "\n".join(p.text for p in paper.pages).lower()
    if analysis and len(analysis.datasets) <= 1:
        found.append(
            ResearchGap(
                title="Limited Cross-Dataset Generalization Evidence",
                description=f"Evaluation appears concentrated on {len(analysis.datasets)} detected dataset(s), leaving cross-domain generalization unverified.",
                supporting_publication=paper.title,
                paper_id=paper.paper_id,
                page_reference="pp. 1-" + str(paper.num_pages),
                classification="inferred",
                category="Evaluation",
                technical_significance="Single-benchmark evaluation can mask distribution shift fragility.",
                potential_improvement="Benchmark across multi-domain or out-of-distribution datasets.",
                feasibility_score=8.0,
                impact_score=8.0,
                implementation_difficulty="Low",
                evidence_quote=f"Detected datasets: {', '.join(d.name for d in analysis.datasets) or 'None explicitly parsed'}",
            )
        )
    if "ablation" not in full_text:
        found.append(
            ResearchGap(
                title="Missing Component Ablation Analysis",
                description="No explicit ablation study was detected in the extracted text to isolate individual module contributions.",
                supporting_publication=paper.title,
                paper_id=paper.paper_id,
                page_reference=f"pp. 1-{paper.num_pages}",
                classification="inferred",
                category="Evaluation",
                technical_significance="Without ablation, the causal contribution of each component remains unclear.",
                potential_improvement="Conduct systematic ablation removing or swapping key architectural modules.",
                feasibility_score=8.5,
                impact_score=7.2,
                implementation_difficulty="Low",
                evidence_quote="Term 'ablation' absent from extracted document text.",
            )
        )
    return found


def build_evidence_to_gap_graph(gaps: list[ResearchGap]) -> nx.Graph:
    """Create a bipartite-style NetworkX graph linking supporting publications -> evidence -> gaps."""
    G = nx.Graph()
    for g in gaps:
        pub_node = f"pub::{g.supporting_publication[:40]}"
        gap_node = g.gap_id
        G.add_node(pub_node, label=g.supporting_publication[:34], full_title=g.supporting_publication, node_type="paper")
        G.add_node(
            gap_node,
            label=g.title[:34],
            full_title=f"{g.title} ({g.classification})",
            node_type="topic" if g.classification == "explicit" else "cited_publication",
        )
        G.add_edge(pub_node, gap_node, relation=f"{g.classification} ({g.page_reference})")
    return G
