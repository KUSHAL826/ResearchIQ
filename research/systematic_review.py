"""Feature 5: Systematic Literature Review Assistant — PRISMA counts, screening, matrix, synthesis."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from pydantic import BaseModel

from core.llm_client import GeminiClient
from core.schemas import PRISMACounts, ParsedPaper, ScholarlyWork


class MatrixExtractionPayload(BaseModel):
    problem: str = ""
    methodology: str = ""
    dataset: str = ""
    results: str = ""
    limitations: str = ""
    contribution: str = ""


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def compute_prisma_counts(
    discovered_works: list[ScholarlyWork],
    parsed_papers: list[ParsedPaper] | None = None,
) -> PRISMACounts:
    """
    Compute logically consistent PRISMA counts strictly from explicit screening decisions.
    Uploaded PDFs are NEVER automatically marked included; every paper tracks its own decision.
    """
    all_records: list[tuple[str, bool, str, bool, str]] = []
    # Tuple: (norm_title, is_duplicate, screening_status, full_text_assessed, exclusion_reason)
    seen_titles: set[str] = set()

    for w in discovered_works:
        nt = _norm_title(w.title)
        if nt:
            seen_titles.add(nt)
        all_records.append(
            (nt, w.is_duplicate, w.screening_status, w.full_text_assessed, w.exclusion_reason)
        )

    for p in parsed_papers or []:
        nt = _norm_title(p.title)
        if nt and nt in seen_titles:
            # Already represented in discovered_works; avoid double-counting unless flagged separately
            continue
        if nt:
            seen_titles.add(nt)
        all_records.append(
            (nt, p.is_duplicate, p.screening_status, p.full_text_assessed, p.exclusion_reason)
        )

    identified = len(all_records)
    duplicates = sum(1 for _, is_dup, _, _, _ in all_records if is_dup)
    unique_records = [rec for rec in all_records if not rec[1]]
    screened = len(unique_records)

    pending_screening = sum(1 for _, _, status, _, _ in unique_records if status == "unscreened")
    needs_review = sum(1 for _, _, status, _, _ in unique_records if status == "needs_review")
    excluded_title_abs = sum(
        1 for _, _, status, ft, _ in unique_records if status == "excluded" and not ft
    )
    included = sum(1 for _, _, status, _, _ in unique_records if status == "included")
    full_text_excluded = sum(
        1 for _, _, status, ft, _ in unique_records if status == "excluded" and ft
    )
    # Every included study and every full-text-assessed record is included in full_text_assessed
    full_text_assessed = sum(
        1 for _, _, status, ft, _ in unique_records if ft or status == "included"
    )

    reasons: Counter[str] = Counter()
    for _, _, status, _, reason in unique_records:
        if status == "excluded":
            clean_reason = (reason or "").strip() or "Criteria mismatch"
            reasons[clean_reason] += 1

    return PRISMACounts(
        records_identified=identified,
        duplicates_removed=duplicates,
        records_screened=screened,
        records_excluded=excluded_title_abs,
        full_text_assessed=full_text_assessed,
        full_text_excluded=full_text_excluded,
        studies_included=included,
        needs_review=needs_review,
        pending_screening=pending_screening,
        exclusion_reasons=dict(reasons),
    )


def apply_rule_based_screening(
    works: list[ScholarlyWork],
    min_year: int | None = None,
    required_keywords: list[str] | None = None,
    exclusion_keywords: list[str] | None = None,
) -> list[ScholarlyWork]:
    """Apply transparent screening rules over non-duplicate works."""
    req = [k.strip().lower() for k in (required_keywords or []) if k.strip()]
    exc = [k.strip().lower() for k in (exclusion_keywords or []) if k.strip()]

    for w in works:
        if w.is_duplicate:
            continue
        blob = f"{w.title} {w.abstract} {' '.join(w.fields_of_study)}".lower()

        if min_year and w.year and w.year < min_year:
            w.screening_status = "excluded"
            w.full_text_assessed = False
            w.exclusion_reason = f"Published before {min_year}"
            continue

        hit_exc = [k for k in exc if k in blob]
        if hit_exc:
            w.screening_status = "excluded"
            w.full_text_assessed = False
            w.exclusion_reason = f"Matched exclusion keyword: {hit_exc[0]}"
            continue

        if req:
            hit_req = [k for k in req if k in blob]
            if len(hit_req) == len(req):
                w.screening_status = "included"
                w.full_text_assessed = True
                w.exclusion_reason = ""
            elif hit_req:
                w.screening_status = "needs_review"
                w.exclusion_reason = ""
            else:
                w.screening_status = "excluded"
                w.full_text_assessed = False
                w.exclusion_reason = "Missing required keywords"
        elif w.screening_status == "unscreened":
            w.screening_status = "needs_review"
    return works


def populate_literature_matrix_row(
    work: ScholarlyWork,
    llm_client: GeminiClient | None = None,
) -> ScholarlyWork:
    """Extract structured matrix fields from a ScholarlyWork abstract using Gemini or transparent heuristic."""
    if llm_client and llm_client.is_configured and work.abstract:
        try:
            prompt = (
                f"Extract concise structured systematic-review matrix fields strictly from this paper metadata.\n"
                f"Title: {work.title}\nYear: {work.year}\nAbstract: {work.abstract}\n"
                f"If a field is not stated in the abstract, write 'Not specified in abstract'."
            )
            res = llm_client.generate_structured(prompt, MatrixExtractionPayload)
            work.extracted_problem = res.problem
            work.extracted_methodology = res.methodology
            work.extracted_dataset = res.dataset
            work.extracted_results = res.results
            work.extracted_limitations = res.limitations
            work.extracted_contribution = res.contribution
            return work
        except Exception:
            pass

    abs_text = work.abstract.strip()
    sentences = re.split(r"(?<=[.!?])\s+", abs_text) if abs_text else []
    work.extracted_problem = sentences[0][:180] if sentences else "Not specified in abstract"
    meth_sents = [s for s in sentences if any(k in s.lower() for k in ("propose", "method", "model", "framework", "algorithm", "architecture"))]
    work.extracted_methodology = meth_sents[0][:180] if meth_sents else (sentences[1][:180] if len(sentences) > 1 else "Not specified in abstract")
    ds_match = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+dataset|\s+benchmark)?)\b", abs_text)
    work.extracted_dataset = ", ".join(dict.fromkeys(ds_match[:3])) if ds_match else "Not specified in abstract"
    res_sents = [s for s in sentences if any(k in s.lower() for k in ("achiev", "accuracy", "outperform", "improv", "result", "%", "score"))]
    work.extracted_results = res_sents[0][:180] if res_sents else "Not specified in abstract"
    work.extracted_limitations = "Not specified in abstract"
    work.extracted_contribution = work.extracted_methodology
    return work


def synthesize_literature_evidence(works: list[ScholarlyWork]) -> dict[str, Any]:
    """Aggregate theme clusters, methodology counts, dataset mentions, and yearly timeline."""
    included = [w for w in works if not w.is_duplicate and w.screening_status == "included"]
    target = included if included else [w for w in works if not w.is_duplicate]

    year_counts: Counter[int] = Counter()
    theme_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()

    for w in target:
        if w.year:
            year_counts[w.year] += 1
        for f in w.fields_of_study:
            theme_counts[f] += 1
        if w.extracted_methodology and w.extracted_methodology != "Not specified in abstract":
            for kw in ("Transformer", "CNN", "GNN", "LLM", "RAG", "Reinforcement Learning", "BERT", "Random Forest", "SVM", "Diffusion"):
                if kw.lower() in w.extracted_methodology.lower() or kw.lower() in w.abstract.lower():
                    method_counts[kw] += 1
        if w.extracted_dataset and w.extracted_dataset != "Not specified in abstract":
            for ds in [d.strip() for d in w.extracted_dataset.split(",") if d.strip()]:
                dataset_counts[ds] += 1

    return {
        "total_synthesized": len(target),
        "included_only": bool(included),
        "year_distribution": dict(sorted(year_counts.items())),
        "theme_distribution": dict(theme_counts.most_common(12)),
        "methodology_distribution": dict(method_counts.most_common(10)),
        "dataset_distribution": dict(dataset_counts.most_common(10)),
    }
