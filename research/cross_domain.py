"""Feature 7: Cross-Domain Research Finder with literature-evidence-backed novelty assessment."""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import CrossDomainIdea, ScholarlyWork
from research.paper_discovery import discover_papers


class CrossDomainIdeaPayload(BaseModel):
    ideas: List[CrossDomainIdea] = Field(default_factory=list)


COMMON_DOMAINS = [
    "Artificial Intelligence",
    "Agriculture",
    "Healthcare",
    "Cybersecurity",
    "Computer Vision",
    "Smart Manufacturing",
    "Climate & Energy",
    "Education",
    "Robotics",
    "Bioinformatics",
    "Finance",
    "Natural Language Processing",
]


def explore_cross_domain(
    domains: list[str],
    llm_client: GeminiClient | None = None,
    per_source_limit: int = 5,
) -> Tuple[list[CrossDomainIdea], list[ScholarlyWork], dict[str, int], list[str]]:
    """
    Search real scholarly literature across the intersection of 2+ selected domains.
    Never claims novelty without checking actual returned literature counts.
    """
    clean_domains = [d.strip() for d in domains if d.strip()]
    if len(clean_domains) < 2:
        return [], [], {}, ["Select at least two research domains."]

    intersection_query = " ".join(clean_domains)
    works, errors = discover_papers(
        intersection_query,
        sources=["openalex", "arxiv"],
        per_source_limit=per_source_limit,
    )
    unique_works = [w for w in works if not w.is_duplicate]

    # Count domain coverage in returned works
    domain_counts: dict[str, int] = {}
    both_count = 0
    for d in clean_domains:
        dl = d.lower()
        domain_counts[d] = sum(
            1 for w in unique_works if dl in f"{w.title} {w.abstract} {' '.join(w.fields_of_study)}".lower()
        )
    for w in unique_works:
        blob = f"{w.title} {w.abstract} {' '.join(w.fields_of_study)}".lower()
        if all(d.lower().split()[0] in blob for d in clean_domains):
            both_count += 1
    domain_counts["Intersection Match"] = both_count

    ideas: list[CrossDomainIdea] = []
    if llm_client and llm_client.is_configured:
        evidence_text = "\n".join(
            f"- [{w.source}, {w.year or 'N/A'}] {w.title} (citations: {w.citation_count if w.citation_count is not None else 'Unknown'})"
            for w in unique_works[:12]
        ) or "No intersection papers returned by APIs."
        prompt = (
            f"Propose 3 interdisciplinary research ideas combining: {', '.join(clean_domains)}.\n"
            f"Retrieved Literature ({len(unique_works)} unique papers, {both_count} direct intersection matches):\n"
            f"{evidence_text}\n\n"
            f"STRICT RULE: Do NOT claim absolute novelty without citing the retrieved literature evidence. "
            f"State in novelty_evidence what the retrieved papers cover and what remains under-explored."
        )
        try:
            payload = llm_client.generate_structured(prompt, CrossDomainIdeaPayload)
            ideas = payload.ideas
            for idea in ideas:
                idea.domains = clean_domains
            return ideas, unique_works, domain_counts, errors
        except Exception:
            pass

    # Grounded fallback ideas strictly referencing fetched titles
    sample_titles = [w.title for w in unique_works[:3]]
    evidence_note = (
        f"Based on {len(unique_works)} retrieved works ({both_count} direct intersection hits). "
        f"Closest existing works: {'; '.join(sample_titles) if sample_titles else 'None returned — needs broader search before claiming novelty'}."
    )
    ideas = [
        CrossDomainIdea(
            title=f"Evidence-Guided {clean_domains[0]} Framework for {clean_domains[1]}",
            domains=clean_domains,
            description=f"Transfer robust representations and uncertainty calibration from {clean_domains[0]} into {clean_domains[1]} workflows.",
            supporting_works=sample_titles,
            novelty_evidence=evidence_note,
            novelty_score=7.5 if both_count < 4 else 5.5,
            feasibility_score=7.5,
            impact_score=8.2,
        ),
        CrossDomainIdea(
            title=f"Benchmarking {' + '.join(clean_domains[:2])} Under Real-World Shift",
            domains=clean_domains,
            description=f"Construct a standardized evaluation benchmark measuring cross-domain reliability across {clean_domains[0]} and {clean_domains[1]}.",
            supporting_works=sample_titles,
            novelty_evidence=evidence_note,
            novelty_score=7.0 if both_count < 5 else 5.0,
            feasibility_score=8.5,
            impact_score=7.8,
        ),
    ]
    return ideas, unique_works, domain_counts, errors
