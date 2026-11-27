"""Feature 7: Cross-Domain Research Finder with rigorous literature overlap & novelty transparency."""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import CrossDomainIdea, ScholarlyWork
from research.paper_discovery import deduplicate_works, discover_papers


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


def _normalize_work_metadata(works: list[ScholarlyWork]) -> list[ScholarlyWork]:
    """Normalize titles, DOIs, venues, and deduplicate authors cleanly."""
    for w in works:
        w.title = re.sub(r"\s+", " ", (w.title or "").strip())
        w.doi = (w.doi or "").replace("https://doi.org/", "").strip().lower()
        w.venue = re.sub(r"\s+", " ", (w.venue or "").strip())
        w.authors = [re.sub(r"\s+", " ", a.strip()) for a in w.authors if a and a.strip()]
    return works


def _domain_matches(work: ScholarlyWork, domain: str) -> bool:
    blob = f"{work.title} {work.abstract} {' '.join(work.fields_of_study)}".lower()
    dl = domain.lower().strip()
    if dl in blob:
        return True
    tokens = [t for t in re.findall(r"[a-z]{3,}", dl) if t not in {"and", "for", "the"}]
    return bool(tokens) and all(t in blob for t in tokens)


def explore_cross_domain(
    domains: list[str],
    llm_client: GeminiClient | None = None,
    per_source_limit: int = 5,
) -> Tuple[list[CrossDomainIdea], list[ScholarlyWork], dict[str, Any], list[str]]:
    """
    Search scholarly literature across selected domains and their intersection.
    Explicitly distinguishes:
    1. Existing research found (intersection matches)
    2. Related research found (single-domain / adjacent studies)
    3. Potentially unexplored intersection areas
    4. Novelty not established (never equates sparse search hits to proven novelty)
    """
    clean_domains = list(dict.fromkeys(d.strip() for d in domains if d.strip()))
    if len(clean_domains) < 2:
        return [], [], {}, ["Select at least two research domains."]

    all_fetched: list[ScholarlyWork] = []
    errors: list[str] = []

    # 1. Intersection search across all selected domains
    intersection_query = " AND ".join(f'"{d}"' if " " in d else d for d in clean_domains)
    inter_works, inter_errs = discover_papers(
        intersection_query,
        sources=["openalex", "arxiv", "semantic_scholar"],
        per_source_limit=per_source_limit,
    )
    all_fetched.extend(inter_works)
    errors.extend(inter_errs)

    # 2. Sample individual domain baselines for overlap context
    for dom in clean_domains[:3]:
        dom_works, dom_errs = discover_papers(
            dom,
            sources=["openalex", "arxiv"],
            per_source_limit=max(2, per_source_limit // 2),
        )
        all_fetched.extend(dom_works)
        errors.extend(dom_errs)

    _normalize_work_metadata(all_fetched)
    deduplicate_works(all_fetched)
    unique_works = [w for w in all_fetched if not w.is_duplicate]

    existing_intersection: list[ScholarlyWork] = []
    related_single_domain: list[ScholarlyWork] = []
    domain_counts: dict[str, int] = {d: 0 for d in clean_domains}

    for w in unique_works:
        matched_domains = [d for d in clean_domains if _domain_matches(w, d)]
        for md in matched_domains:
            domain_counts[md] += 1
        if len(matched_domains) >= 2:
            existing_intersection.append(w)
        elif len(matched_domains) == 1 or w.abstract:
            related_single_domain.append(w)

    novelty_established = False  # Establishing true novelty requires exhaustive systematic review
    if len(existing_intersection) >= 5:
        novelty_status = "Established intersection literature exists — novelty requires narrow sub-problem differentiation."
    elif len(existing_intersection) > 0:
        novelty_status = "Partial intersection literature found; novelty is NOT established without broader multi-query review."
    else:
        novelty_status = "Novelty not established: zero direct hits in this quick query may reflect search phrasing rather than an unexplored field."

    evidence_breakdown: dict[str, Any] = {
        "counts": {
            **domain_counts,
            "Existing Intersection": len(existing_intersection),
            "Related Single-Domain": len(related_single_domain),
        },
        "existing_research": [w.title for w in existing_intersection[:6]],
        "related_research": [w.title for w in related_single_domain[:6]],
        "potentially_unexplored": [
            f"Cross-domain benchmark transfer between {clean_domains[0]} and {clean_domains[1]}",
            f"Failure-mode & safety auditing when applying {clean_domains[0]} methods to {clean_domains[1]} data",
        ],
        "novelty_status": novelty_status,
        "novelty_established": novelty_established,
        "total_unique_examined": len(unique_works),
    }

    ideas: list[CrossDomainIdea] = []
    if llm_client and llm_client.is_configured:
        existing_txt = "\n".join(f"- {w.title} ({w.year or 'n.d.'}, {w.source})" for w in existing_intersection[:8]) or "None matched directly."
        related_txt = "\n".join(f"- {w.title} ({w.year or 'n.d.'}, {w.source})" for w in related_single_domain[:8]) or "None."
        prompt = (
            f"Propose 3 interdisciplinary research directions combining: {', '.join(clean_domains)}.\n"
            f"1) Existing Direct Intersection Research ({len(existing_intersection)} found):\n{existing_txt}\n"
            f"2) Related Single-Domain Research ({len(related_single_domain)} found):\n{related_txt}\n\n"
            f"STRICT RULES:\n"
            f"- NEVER claim a topic is definitively novel just because few papers were returned.\n"
            f"- Explicitly cite which existing or related papers were found and state what remains unverified.\n"
            f"- Keep novelty_score conservative (<= 6.5 unless clear gap evidence is cited)."
        )
        try:
            payload = llm_client.generate_structured(prompt, CrossDomainIdeaPayload)
            ideas = payload.ideas
            for idea in ideas:
                idea.domains = clean_domains
            return ideas, unique_works, evidence_breakdown, errors
        except Exception:
            pass

    sample_existing = [w.title for w in existing_intersection[:3]]
    sample_related = [w.title for w in related_single_domain[:3]]
    evidence_note = (
        f"{novelty_status} Existing intersection hits: {len(existing_intersection)}; "
        f"related single-domain hits: {len(related_single_domain)}. "
        f"Supporting titles: {'; '.join(sample_existing or sample_related) or 'None retrieved — run broader queries'}."
    )
    ideas = [
        CrossDomainIdea(
            title=f"Domain-Adapted {clean_domains[0]} Evaluation in {clean_domains[1]}",
            domains=clean_domains,
            description=f"Evaluate transferability, calibration, and failure modes when adapting {clean_domains[0]} architectures to {clean_domains[1]} benchmarks.",
            supporting_works=sample_existing or sample_related,
            novelty_evidence=evidence_note,
            novelty_score=5.0,
            feasibility_score=7.5,
            impact_score=7.8,
        ),
        CrossDomainIdea(
            title=f"Standardized Benchmark for {' + '.join(clean_domains[:2])} Integration",
            domains=clean_domains,
            description=f"Build a reproducible evaluation protocol bridging {clean_domains[0]} metrics and {clean_domains[1]} domain constraints.",
            supporting_works=sample_related or sample_existing,
            novelty_evidence=evidence_note,
            novelty_score=5.0,
            feasibility_score=8.0,
            impact_score=7.5,
        ),
    ]
    return ideas, unique_works, evidence_breakdown, errors
