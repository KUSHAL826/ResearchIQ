"""Part 8: Publication Strategy & Suitability Intelligence (OpenAlex real metadata + explicit verification caveats)."""

from __future__ import annotations

from core.schemas import PublicationSuitabilityItem, ResearchProject
from paper_studio.publisher_exporter import discover_target_journals


def build_publication_suitability_report(
    project: ResearchProject,
    query_override: str = "",
    limit: int = 8,
) -> tuple[list[PublicationSuitabilityItem], str | None]:
    """
    Discover candidate venues via OpenAlex /sources and produce a suitability report.
    Never guarantees acceptance, indexing status, or venue legitimacy — marks verification required.
    """
    q = (query_override or project.selected_topic or project.domain).strip()
    journals, err = discover_target_journals(q, limit=limit)
    if err:
        return [], err

    fmt = project.publication_prep.target_format
    items: list[PublicationSuitabilityItem] = []
    for j in journals:
        vtype = str(j.get("type", "journal"))
        issn = j.get("issn", "N/A")
        works_cnt = j.get("works_count")
        cites_cnt = j.get("cited_by_count")
        items.append(
            PublicationSuitabilityItem(
                venue_name=j.get("name", "Unknown Venue"),
                publisher_or_type=f"{vtype.title()} | {j.get('is_oa', 'Unknown OA status')}",
                scope_match=f"Matched query '{q}' via OpenAlex metadata (works={works_cnt}, cited_by={cites_cnt}).",
                indexing_note=f"ISSN: {issn} — verify active indexing (Scopus/WoS/DBLP) on official publisher site.",
                submission_format=f"Draft ready in {fmt}; verify official venue template (.cls/.dotx).",
                word_or_page_limit="Verify exact page/word limit on official Call for Papers / Author Guidelines.",
                deadlines_note="Check official conference/journal portal for current submission deadlines.",
                verification_required=True,
                homepage=j.get("homepage", ""),
            )
        )

    project.publication_suitability = items
    return items, None
