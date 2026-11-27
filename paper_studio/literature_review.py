"""Studio Stages 3 & 4: Literature Review Synthesis and Gap-to-Contribution Mapping."""

from __future__ import annotations

from typing import Any

from core.schemas import ResearchProject


def build_existing_vs_proposed_matrix(project: ResearchProject) -> list[dict[str, Any]]:
    """Build a structured comparison matrix contrasting existing gaps/limitations with proposed improvements."""
    rows: list[dict[str, Any]] = []
    for g in project.research_gaps:
        rows.append(
            {
                "Existing Work / Source": f"{g.supporting_publication[:45]} ({g.page_reference})",
                "Identified Gap / Limitation": g.title,
                "Type": g.classification.title(),
                "Proposed Contribution": g.potential_improvement or "Targeted architectural & experimental improvement",
                "Impact": g.impact_score,
                "Feasibility": g.feasibility_score,
            }
        )
    if not rows:
        included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
        for w in included[:6]:
            rows.append(
                {
                    "Existing Work / Source": f"{w.title[:45]} ({w.year or 'n.d.'})",
                    "Identified Gap / Limitation": w.extracted_limitations or "Requires full-text limitation extraction",
                    "Type": "Literature",
                    "Proposed Contribution": project.selected_topic or "Proposed methodology extension",
                    "Impact": 7.0,
                    "Feasibility": 7.0,
                }
            )
    return rows
