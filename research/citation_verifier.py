"""Feature 9: Citation Claim Verifier — prioritizes cited sources and never treats lexical overlap as proof."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import (
    ClaimVerificationResult,
    ManuscriptSectionState,
    ParsedPaper,
    ScholarlyWork,
)
from core.vector_store import SessionVectorStore

CITATION_SENTENCE_RE = re.compile(
    r"([^.\n]{20,300}(?:\[[0-9,\s\-]+\]|\([A-Z][A-Za-z\s.&]+,\s*(?:19|20)\d{2}\)|et al\.)[^.\n]*\.)"
)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
NEGATION_WORDS = {"not", "never", "no", "fail", "failed", "fails", "cannot", "without", "inferior", "worse", "underperform"}


class ClaimJudgementPayload(BaseModel):
    support_status: str  # Supported | Partially supported | Not supported | Unable to verify
    contradiction_note: str = ""
    reasoning: str = ""
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


def extract_claims_from_manuscript(sections: dict[str, ManuscriptSectionState]) -> list[tuple[str, str, str]]:
    """Extract (section_name, claim_sentence, cited_source_marker) tuples."""
    claims: list[tuple[str, str, str]] = []
    for sec_name, sec in sections.items():
        if sec_name == "References" or not sec.content.strip():
            continue
        for match in CITATION_SENTENCE_RE.finditer(sec.content):
            sent = re.sub(r"\s+", " ", match.group(1)).strip()
            cite_m = re.search(r"(\[[0-9,\s\-]+\]|\([A-Z][A-Za-z\s.&]+,\s*(?:19|20)\d{2}\))", sent)
            marker = cite_m.group(1) if cite_m else "Citation marker"
            claims.append((sec_name, sent, marker))
    return claims[:30]


def _resolve_cited_targets(
    cited_source: str,
    parsed_papers: list[ParsedPaper] | None,
    discovered_works: list[ScholarlyWork] | None,
    references_text: str = "",
) -> tuple[set[str], set[str], str]:
    """
    Resolve a citation hint (e.g. '[1]', 'Smith et al., 2023', or paper title)
    to matching paper_ids and work_ids so evidence from the cited source is prioritized.
    """
    target_paper_ids: set[str] = set()
    target_work_ids: set[str] = set()
    hint = (cited_source or "").strip()
    if not hint:
        return target_paper_ids, target_work_ids, ""

    resolved_hint = hint
    # Resolve numeric citation index '[1]' against References section lines if available
    idx_match = re.match(r"^\[(\d+)\]$", hint)
    if idx_match and references_text.strip():
        ref_num = int(idx_match.group(1))
        ref_lines = [ln.strip() for ln in references_text.splitlines() if ln.strip()]
        if 1 <= ref_num <= len(ref_lines):
            resolved_hint = ref_lines[ref_num - 1]

    hint_tokens = set(re.findall(r"[a-z0-9]{3,}", resolved_hint.lower())) - {"etal", "doi", "http", "https"}
    if not hint_tokens:
        return target_paper_ids, target_work_ids, resolved_hint

    for p in parsed_papers or []:
        meta_blob = f"{p.title} {' '.join(p.authors)} {p.year or ''} {p.doi}".lower()
        meta_tokens = set(re.findall(r"[a-z0-9]{3,}", meta_blob))
        if len(hint_tokens & meta_tokens) >= min(2, len(hint_tokens)):
            target_paper_ids.add(p.paper_id)

    for w in discovered_works or []:
        meta_blob = f"{w.title} {' '.join(w.authors)} {w.year or ''} {w.doi}".lower()
        meta_tokens = set(re.findall(r"[a-z0-9]{3,}", meta_blob))
        if len(hint_tokens & meta_tokens) >= min(2, len(hint_tokens)):
            target_work_ids.add(w.work_id)

    return target_paper_ids, target_work_ids, resolved_hint


def _detect_offline_contradiction(claim_text: str, passage: str) -> str:
    """
    Conservative offline contradiction detector: checks for conflicting numeric measurements
    on shared metrics or opposing negation polarity on shared key entities.
    """
    claim_lower = claim_text.lower()
    pass_lower = passage.lower()

    claim_nums = set(NUMBER_RE.findall(claim_text))
    pass_nums = set(NUMBER_RE.findall(passage))
    shared_metrics = [
        m for m in ("accuracy", "f1", "precision", "recall", "auc", "rmse", "latency")
        if m in claim_lower and m in pass_lower
    ]
    if shared_metrics and claim_nums and pass_nums and not (claim_nums & pass_nums):
        return (
            f"Claim cites numeric value(s) {sorted(claim_nums)} for {', '.join(shared_metrics)}, "
            f"whereas passage reports {sorted(pass_nums)}."
        )

    claim_neg = bool(set(re.findall(r"[a-z]+", claim_lower)) & NEGATION_WORDS)
    pass_neg = bool(set(re.findall(r"[a-z]+", pass_lower)) & NEGATION_WORDS)
    if claim_neg != pass_neg:
        claim_terms = set(re.findall(r"[a-z]{4,}", claim_lower)) - NEGATION_WORDS
        pass_terms = set(re.findall(r"[a-z]{4,}", pass_lower)) - NEGATION_WORDS
        if len(claim_terms & pass_terms) >= 3:
            return "Polarity difference detected (negation mismatch on shared technical terms)."
    return ""


def verify_claim_against_evidence(
    claim_text: str,
    cited_source: str,
    section_name: str,
    vector_store: SessionVectorStore,
    discovered_works: list[ScholarlyWork] | None = None,
    llm_client: GeminiClient | None = None,
    parsed_papers: list[ParsedPaper] | None = None,
    references_text: str = "",
) -> ClaimVerificationResult:
    """
    Verify a claim against indexed PDF chunks and discovered abstracts.
    - Prioritizes evidence from the cited source when identifiable.
    - NEVER marks a claim Supported on DOI existence or lexical similarity alone.
    - When Gemini is unavailable, uses a conservative fallback returning Unable to verify
      unless a direct contradiction is detected.
    """
    target_paper_ids, target_work_ids, resolved_hint = _resolve_cited_targets(
        cited_source, parsed_papers, discovered_works, references_text
    )

    candidates: list[tuple[str, str, str, float, bool]] = []  # (passage, title, page_ref, score, is_cited_match)
    paper_title_map = {p.paper_id: p.title for p in (parsed_papers or [])}

    if vector_store and vector_store.chunks:
        # First prioritize cited paper(s) if resolved
        for pid in target_paper_ids:
            hits = vector_store.search(claim_text, top_k=3, llm_client=llm_client, paper_id_filter=pid)
            for chunk, score in hits:
                title = paper_title_map.get(chunk.paper_id, chunk.paper_title)
                candidates.append(
                    (chunk.text, title, f"pp. {chunk.page_start}-{chunk.page_end}", score, True)
                )
        # Global search across all indexed PDFs
        global_hits = vector_store.search(claim_text, top_k=4, llm_client=llm_client)
        for chunk, score in global_hits:
            is_match = chunk.paper_id in target_paper_ids
            title = paper_title_map.get(chunk.paper_id, chunk.paper_title)
            candidates.append(
                (chunk.text, title, f"pp. {chunk.page_start}-{chunk.page_end}", score, is_match)
            )

    claim_tokens = set(re.findall(r"[a-z0-9]{3,}", claim_text.lower()))
    for w in discovered_works or []:
        if not w.abstract.strip():
            continue
        abs_tokens = set(re.findall(r"[a-z0-9]{3,}", w.abstract.lower()))
        if not abs_tokens or not claim_tokens:
            continue
        overlap = len(claim_tokens & abs_tokens) / max(1, len(claim_tokens))
        is_match = w.work_id in target_work_ids
        if is_match or overlap > 0.12:
            candidates.append((w.abstract[:800], w.title, f"Abstract ({w.source})", overlap, is_match))

    # Deduplicate and sort prioritizing cited-source matches first, then alignment score
    seen_passages: set[str] = set()
    unique_candidates: list[tuple[str, str, str, float, bool]] = []
    for item in candidates:
        key = item[0][:120]
        if key not in seen_passages:
            seen_passages.add(key)
            unique_candidates.append(item)

    unique_candidates.sort(key=lambda x: (1 if x[4] else 0, x[3]), reverse=True)

    if not unique_candidates:
        return ClaimVerificationResult(
            section_name=section_name,
            claim_text=claim_text,
            cited_source=cited_source or "Unspecified",
            support_status="Unable to verify",
            evidence_passage="No matching source passage found in uploaded PDFs or retrieved abstracts.",
            page_reference="N/A",
            reasoning="Verification requires accessible source passages; DOI existence or citation label alone is insufficient.",
            confidence_score=0.0,
            confidence_label="Insufficient evidence — indicator only, not proof",
        )

    best_passage, best_title, best_page, best_score, is_cited_match = unique_candidates[0]
    cited_display = f"{cited_source} -> {best_title}" if cited_source else best_title

    if llm_client and llm_client.is_configured:
        top_ev = "\n\n".join(
            f"[Candidate {i+1} | {'CITED SOURCE MATCH' if c[4] else 'Session Source'} | {c[1]} | {c[2]}]\n{c[0][:700]}"
            for i, c in enumerate(unique_candidates[:3])
        )
        try:
            prompt = (
                f"Carefully verify whether the retrieved source evidence factually supports the claim.\n"
                f"Claim: {claim_text}\n"
                f"Cited Marker: {cited_source} (Resolved hint: {resolved_hint or 'Unresolved'})\n\n"
                f"RETRIEVED PASSAGES:\n{top_ev}\n\n"
                f"RULES:\n"
                f"- 'Supported': direct factual alignment with the passage(s).\n"
                f"- 'Partially supported': some aspects match, others are missing or overstated.\n"
                f"- 'Not supported': passage directly contradicts the claim (explain in contradiction_note).\n"
                f"- 'Unable to verify': evidence is ambiguous, off-topic, or insufficient.\n"
                f"- Do NOT confuse shared terminology with factual proof."
            )
            judgement = llm_client.generate_structured(prompt, ClaimJudgementPayload)
            allowed = {"Supported", "Partially supported", "Not supported", "Unable to verify"}
            status = judgement.support_status if judgement.support_status in allowed else "Unable to verify"
            return ClaimVerificationResult(
                section_name=section_name,
                claim_text=claim_text,
                cited_source=cited_display,
                support_status=status,  # type: ignore[arg-type]
                evidence_passage=best_passage[:550],
                page_reference=best_page,
                contradiction_note=judgement.contradiction_note,
                reasoning=judgement.reasoning,
                confidence_score=round(float(judgement.confidence_score), 2),
                confidence_label="AI assessment indicator — inspect passage before final submission",
            )
        except Exception:
            pass

    # Conservative offline fallback: never mark 'Supported' based on lexical similarity alone
    contradiction = _detect_offline_contradiction(claim_text, best_passage)
    bounded_score = round(min(0.99, max(0.0, float(best_score))), 2)
    if contradiction:
        return ClaimVerificationResult(
            section_name=section_name,
            claim_text=claim_text,
            cited_source=cited_display,
            support_status="Not supported",
            evidence_passage=best_passage[:550],
            page_reference=best_page,
            contradiction_note=contradiction,
            reasoning="Offline contradiction check flagged a conflict between the claim and retrieved passage.",
            confidence_score=bounded_score,
            confidence_label="Heuristic contradiction flag — manual verification required",
        )

    source_note = "prioritized cited source" if is_cited_match else "closest session passage"
    return ClaimVerificationResult(
        section_name=section_name,
        claim_text=claim_text,
        cited_source=cited_display,
        support_status="Unable to verify",
        evidence_passage=best_passage[:550],
        page_reference=best_page,
        reasoning=(
            f"Conservative offline fallback: retrieved {source_note} (alignment indicator={bounded_score:.2f}), "
            f"but lexical similarity cannot prove factual support without Gemini semantic verification or human review."
        ),
        confidence_score=bounded_score,
        confidence_label="Retrieval similarity indicator only — not factual proof",
    )
