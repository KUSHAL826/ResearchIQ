"""Studio Stages 10 & 11: Manuscript Section Management, Health & Reference Completeness."""

from __future__ import annotations

import re
from typing import Any

from core.config import MANUSCRIPT_SECTIONS
from core.schemas import ManuscriptSectionState, ResearchProject
from research.consistency_checker import run_consistency_check


def compute_manuscript_completion(project: ResearchProject) -> dict[str, Any]:
    """Compute completion and approval metrics from real manuscript section states."""
    total = len(MANUSCRIPT_SECTIONS)
    drafted = 0
    approved = 0
    total_words = 0
    section_rows: list[dict[str, Any]] = []

    for name in MANUSCRIPT_SECTIONS:
        sec = project.manuscript_sections.get(name, ManuscriptSectionState(name=name))
        words = len(sec.content.split()) if sec.content.strip() else 0
        sec.word_count = words
        if words > 0:
            drafted += 1
        if sec.approved:
            approved += 1
        total_words += words
        section_rows.append(
            {
                "Section": name,
                "Words": words,
                "Drafted": "Yes" if words > 0 else "No",
                "Approved": "Approved" if sec.approved else "Pending",
            }
        )

    pct = round(((drafted * 0.5 + approved * 0.5) / max(1, total)) * 100.0, 1)
    return {
        "total_sections": total,
        "drafted_count": drafted,
        "approved_count": approved,
        "total_words": total_words,
        "completion_pct": pct,
        "section_rows": section_rows,
    }


def check_reference_completeness(project: ResearchProject) -> dict[str, Any]:
    """Check whether in-text citations match the References section and session bibliography."""
    refs_sec = project.manuscript_sections.get("References")
    refs_text = refs_sec.content if refs_sec else ""

    cited_numbers: set[int] = set()
    for name, sec in project.manuscript_sections.items():
        if name == "References" or not sec.content:
            continue
        for match in re.findall(r"\[(\d+)\]", sec.content):
            cited_numbers.add(int(match))

    ref_lines = [ln.strip() for ln in refs_text.splitlines() if ln.strip()]
    available_count = len(ref_lines)
    missing_refs = sorted(n for n in cited_numbers if n > available_count or n < 1)

    incomplete_metadata: list[str] = []
    for w in project.discovered_works:
        if not w.is_duplicate and w.screening_status == "included":
            if not w.authors or not w.year:
                incomplete_metadata.append(w.title[:60])

    return {
        "in_text_citation_indices": sorted(cited_numbers),
        "reference_entries_count": available_count,
        "missing_reference_indices": missing_refs,
        "incomplete_metadata_titles": incomplete_metadata,
        "is_complete": len(missing_refs) == 0 and available_count > 0,
    }


def compute_manuscript_health(project: ResearchProject) -> dict[str, Any]:
    """Aggregate Stage 11 validation dashboard metrics."""
    comp = compute_manuscript_completion(project)
    consistency = run_consistency_check(project)
    ref_check = check_reference_completeness(project)

    verified = sum(1 for c in project.claim_verifications if c.support_status == "Supported")
    partial = sum(1 for c in project.claim_verifications if c.support_status == "Partially supported")
    unsupported = sum(1 for c in project.claim_verifications if c.support_status in ("Not supported", "Unable to verify"))

    return {
        "completion": comp,
        "consistency": consistency,
        "reference_check": ref_check,
        "claim_counts": {
            "Supported": verified,
            "Partially supported": partial,
            "Unverified / Not supported": unsupported,
        },
    }
