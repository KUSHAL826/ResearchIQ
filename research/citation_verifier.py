"""Feature 9: Citation Claim Verifier — validates manuscript claims against actual source passages."""

from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel

from core.llm_client import GeminiClient
from core.schemas import ClaimVerificationResult, ManuscriptSectionState, ScholarlyWork
from core.vector_store import SessionVectorStore

CITATION_SENTENCE_RE = re.compile(
    r"([^.\n]{20,300}(?:\[[0-9,\s\-]+\]|\([A-Z][A-Za-z\s.&]+,\s*(?:19|20)\d{2}\)|et al\.)[^.\n]*\.)"
)


class ClaimJudgementPayload(BaseModel):
    support_status: str  # Supported | Partially supported | Not supported | Unable to verify
    contradiction_note: str = ""
    reasoning: str = ""


def extract_claims_from_manuscript(sections: dict[str, ManuscriptSectionState]) -> list[tuple[str, str, str]]:
    """Extract (section_name, claim_sentence, cited_source_marker) tuples."""
    claims: list[tuple[str, str, str]] = []
    for sec_name, sec in sections.items():
        if not sec.content.strip():
            continue
        for match in CITATION_SENTENCE_RE.finditer(sec.content):
            sent = re.sub(r"\s+", " ", match.group(1)).strip()
            cite_m = re.search(r"(\[[0-9,\s\-]+\]|\([A-Z][A-Za-z\s.&]+,\s*(?:19|20)\d{2}\))", sent)
            marker = cite_m.group(1) if cite_m else "Citation marker"
            claims.append((sec_name, sent, marker))
    return claims[:30]


def verify_claim_against_evidence(
    claim_text: str,
    cited_source: str,
    section_name: str,
    vector_store: SessionVectorStore,
    discovered_works: list[ScholarlyWork] | None = None,
    llm_client: GeminiClient | None = None,
) -> ClaimVerificationResult:
    """
    Verify a claim against indexed PDF chunks and discovered abstracts.
    NEVER marks a claim Supported just because a DOI exists — requires textual passage alignment.
    """
    candidates: list[tuple[str, str, str, float]] = []  # (passage, source_title, page_ref, score)

    if vector_store and vector_store.chunks:
        hits = vector_store.search(claim_text, top_k=3, llm_client=llm_client)
        for chunk, score in hits:
            candidates.append(
                (chunk.text, chunk.paper_title, f"pp. {chunk.page_start}-{chunk.page_end}", score)
            )

    # Also check discovered works' abstracts (explicitly NOT verified by DOI existence alone)
    claim_tokens = set(re.findall(r"[a-z0-9]{3,}", claim_text.lower()))
    for w in discovered_works or []:
        if not w.abstract.strip():
            continue
        abs_tokens = set(re.findall(r"[a-z0-9]{3,}", w.abstract.lower()))
        if not abs_tokens or not claim_tokens:
            continue
        overlap = len(claim_tokens & abs_tokens) / max(1, len(claim_tokens))
        if overlap > 0.12:
            candidates.append((w.abstract[:700], w.title, f"Abstract ({w.source})", overlap))

    candidates.sort(key=lambda x: x[3], reverse=True)

    if not candidates:
        return ClaimVerificationResult(
            section_name=section_name,
            claim_text=claim_text,
            cited_source=cited_source or "Unspecified",
            support_status="Unable to verify",
            evidence_passage="No matching source passage found in uploaded PDFs or retrieved abstracts.",
            page_reference="N/A",
            reasoning="Verification requires accessible source text; DOI existence alone is insufficient.",
        )

    best_passage, best_title, best_page, best_score = candidates[0]

    if llm_client and llm_client.is_configured:
        try:
            prompt = (
                f"Evaluate whether the following retrieved source passage supports the manuscript claim.\n"
                f"Claim: {claim_text}\n"
                f"Cited Marker: {cited_source}\n"
                f"Source ({best_title}, {best_page}): {best_passage}\n\n"
                f"Return support_status strictly as one of: 'Supported', 'Partially supported', 'Not supported', 'Unable to verify'."
            )
            judgement = llm_client.generate_structured(prompt, ClaimJudgementPayload)
            allowed = {"Supported", "Partially supported", "Not supported", "Unable to verify"}
            status = judgement.support_status if judgement.support_status in allowed else "Unable to verify"
            return ClaimVerificationResult(
                section_name=section_name,
                claim_text=claim_text,
                cited_source=f"{cited_source} -> {best_title}" if cited_source else best_title,
                support_status=status,  # type: ignore[arg-type]
                evidence_passage=best_passage[:500],
                page_reference=best_page,
                contradiction_note=judgement.contradiction_note,
                reasoning=judgement.reasoning,
            )
        except Exception:
            pass

    # Transparent lexical alignment classification when offline
    if best_score >= 0.45:
        status = "Supported"
        reason = f"High passage token/semantic alignment (score={best_score:.2f})."
    elif best_score >= 0.24:
        status = "Partially supported"
        reason = f"Partial lexical alignment with source passage (score={best_score:.2f}); manual inspection recommended."
    else:
        status = "Unable to verify"
        reason = f"Low alignment score ({best_score:.2f}) with available passages."

    return ClaimVerificationResult(
        section_name=section_name,
        claim_text=claim_text,
        cited_source=f"{cited_source} -> {best_title}" if cited_source else best_title,
        support_status=status,  # type: ignore[arg-type]
        evidence_passage=best_passage[:500],
        page_reference=best_page,
        reasoning=reason,
    )
