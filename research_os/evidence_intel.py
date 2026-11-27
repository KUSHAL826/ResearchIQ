"""Part 2.2: Centralized Research Evidence Intelligence (traceable passages & classifications)."""

from __future__ import annotations

from core.llm_client import GeminiClient
from core.schemas import EvidenceItem, ResearchProject
from core.vector_store import SessionVectorStore


def collect_project_evidence(
    project: ResearchProject,
    vector_store: SessionVectorStore | None = None,
    query: str = "",
    llm_client: GeminiClient | None = None,
) -> list[EvidenceItem]:
    """
    Aggregate traceable evidence across uploaded PDFs (FAISS passages),
    extracted limitations, and claim verification records.
    Never fabricates page numbers or passages.
    """
    items: list[EvidenceItem] = []

    # 1. Live semantic retrieval over indexed PDFs when query is supplied
    if query.strip() and vector_store and vector_store.chunks:
        hits = vector_store.search(query.strip(), top_k=6, llm_client=llm_client)
        for chunk, score in hits:
            items.append(
                EvidenceItem(
                    statement=f"Retrieved passage matching '{query.strip()[:50]}'",
                    source_paper=chunk.paper_title,
                    paper_id=chunk.paper_id,
                    page_number=f"pp. {chunk.page_start}-{chunk.page_end}",
                    section=chunk.section,
                    passage=chunk.text[:600],
                    retrieval_score=round(float(score), 3),
                    classification="Direct evidence" if score >= 0.45 else "Supporting evidence",
                )
            )

    # 2. Evidence from Paper Intelligence limitations & results
    for pid, analysis in project.paper_analyses.items():
        for lim in analysis.limitations[:4]:
            items.append(
                EvidenceItem(
                    statement=lim.description,
                    source_paper=analysis.paper_title,
                    paper_id=pid,
                    page_number=lim.page_reference or "Page not parsed",
                    section="Limitations / Discussion",
                    passage=lim.source_quote or lim.description,
                    retrieval_score=1.0,
                    classification="Direct evidence",
                )
            )

    # 3. Evidence from Research Gaps
    for g in project.research_gaps:
        items.append(
            EvidenceItem(
                statement=f"{g.title}: {g.description}",
                source_paper=g.supporting_publication,
                paper_id=g.paper_id,
                page_number=g.page_reference,
                section="Gap Evidence",
                passage=g.evidence_quote or g.description,
                retrieval_score=0.9 if g.classification == "explicit" else 0.6,
                classification="Direct evidence" if g.classification == "explicit" else "Indirect evidence",
            )
        )

    # 4. Evidence from Claim Verifications
    status_map = {
        "Supported": "Direct evidence",
        "Partially supported": "Supporting evidence",
        "Not supported": "Contradictory evidence",
        "Unable to verify": "Insufficient evidence",
    }
    for clm in project.claim_verifications:
        items.append(
            EvidenceItem(
                statement=clm.claim_text,
                source_paper=clm.cited_source,
                page_number=clm.page_reference or "N/A",
                section=clm.section_name or "Manuscript",
                passage=clm.evidence_passage,
                retrieval_score=clm.confidence_score,
                classification=status_map.get(clm.support_status, "Insufficient evidence"),  # type: ignore[arg-type]
            )
        )

    project.evidence_registry = items
    return items
