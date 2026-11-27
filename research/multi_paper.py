"""Feature 3: Multi-Paper Intelligence — methodology/dataset/performance matrix, contradictions & gap overlap."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import (
    ContradictionItem,
    MultiPaperComparison,
    PaperAnalysis,
    ParsedPaper,
    ResearchGap,
)


class ContradictionListPayload(BaseModel):
    common_themes: list[str] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    synthesis_summary: str = ""


def compare_papers(
    papers: list[ParsedPaper],
    analyses: dict[str, PaperAnalysis],
    gaps: list[ResearchGap] | None = None,
    llm_client: GeminiClient | None = None,
) -> MultiPaperComparison:
    """Compare multiple selected papers across methodology, algorithms, datasets, metrics, themes, and gaps."""
    if not papers:
        return MultiPaperComparison()

    gaps = gaps or []
    matrix: list[dict[str, Any]] = []
    theme_counter: Counter[str] = Counter()

    for p in papers:
        a = analyses.get(p.paper_id)
        algs = ", ".join(alg.name for alg in a.algorithms) if a and a.algorithms else "Not extracted"
        ds = ", ".join(d.name for d in a.datasets) if a and a.datasets else "Not extracted"
        res = "; ".join(f"{r.metric}: {r.raw_value or r.value}" for r in a.results[:4]) if a and a.results else "Not extracted"
        lims = "; ".join(l.description[:90] for l in a.limitations[:2]) if a and a.limitations else "Not extracted"
        matrix.append(
            {
                "paper_id": p.paper_id,
                "Title": p.title[:65],
                "Year": p.year or "N/A",
                "Pages": p.num_pages,
                "Methodology": " -> ".join(a.methodology_steps[:3]) if a and a.methodology_steps else p.hierarchical_summary[:120],
                "Algorithms": algs,
                "Datasets": ds,
                "Reported Results": res,
                "Key Limitations": lims,
            }
        )
        if a:
            for alg in a.algorithms:
                theme_counter[alg.name] += 1
            for d in a.datasets:
                theme_counter[d.name] += 1

    common_themes = [k for k, c in theme_counter.most_common(10) if c >= 1]
    contradictions: list[ContradictionItem] = []
    synthesis = ""

    if len(papers) >= 2 and llm_client and llm_client.is_configured:
        try:
            summaries = "\n\n".join(
                f"PAPER: {p.title} ({p.year or 'N/A'})\n"
                f"Problem: {analyses[p.paper_id].problem_statement if p.paper_id in analyses else p.hierarchical_summary}\n"
                f"Algorithms: {', '.join(x.name for x in analyses[p.paper_id].algorithms) if p.paper_id in analyses else 'N/A'}\n"
                f"Results: {'; '.join(f'{r.metric}={r.raw_value}' for r in analyses[p.paper_id].results[:5]) if p.paper_id in analyses else 'N/A'}"
                for p in papers[:6]
            )
            prompt = (
                f"Compare these research papers. Identify common_themes, any genuine methodological or empirical "
                f"contradictions (only if supported by the summaries), and a concise synthesis_summary.\n\n{summaries}"
            )
            payload = llm_client.generate_structured(prompt, ContradictionListPayload)
            if payload.common_themes:
                common_themes = payload.common_themes
            contradictions = payload.contradictions
            synthesis = payload.synthesis_summary
        except Exception:
            pass

    if not synthesis:
        synthesis = (
            f"Compared {len(papers)} paper(s). Shared technical concepts detected: "
            f"{', '.join(common_themes[:5]) if common_themes else 'None overlapping yet'}."
        )

    # Gap overlap across papers by category
    selected_ids = {p.paper_id for p in papers}
    selected_gaps = [g for g in gaps if g.paper_id in selected_ids]
    cat_papers: dict[str, set[str]] = {}
    for g in selected_gaps:
        cat_papers.setdefault(g.category, set()).add(g.supporting_publication[:45])
    gap_overlaps = [
        {"Category": cat, "Paper Count": len(pubs), "Publications": ", ".join(sorted(pubs))}
        for cat, pubs in cat_papers.items()
    ]

    return MultiPaperComparison(
        selected_paper_ids=[p.paper_id for p in papers],
        common_themes=common_themes,
        methodology_matrix=matrix,
        contradictions=contradictions,
        gap_overlaps=gap_overlaps,
        synthesis_summary=synthesis,
    )
