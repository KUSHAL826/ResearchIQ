"""Section-aware academic paper parser mapping headings to canonical paper structure."""

from __future__ import annotations

import re

from core.schemas import PageContent, SectionBlock

CANONICAL_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Abstract", ("abstract", "summary")),
    ("Introduction", ("introduction", "background", "overview", "motivation")),
    (
        "Literature Review",
        (
            "literature review",
            "related work",
            "related works",
            "prior work",
            "state of the art",
            "previous work",
            "survey",
        ),
    ),
    (
        "Methodology",
        (
            "methodology",
            "methods",
            "method",
            "proposed approach",
            "proposed method",
            "proposed system",
            "proposed framework",
            "architecture",
            "approach",
            "materials and methods",
            "model formulation",
        ),
    ),
    (
        "Experiments",
        (
            "experiments",
            "experimental setup",
            "experimental evaluation",
            "implementation",
            "datasets",
            "evaluation setup",
            "setup",
        ),
    ),
    (
        "Results",
        (
            "results",
            "results and discussion",
            "discussion",
            "experimental results",
            "evaluation",
            "performance analysis",
            "ablation",
        ),
    ),
    (
        "Conclusion",
        (
            "conclusion",
            "conclusions",
            "conclusion and future work",
            "concluding remarks",
            "future work",
            "limitations",
        ),
    ),
]

LINE_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+)?([A-Za-z][A-Za-z0-9\s,&:\-/]{2,70})\s*$"
)


def classify_heading(heading_text: str) -> str | None:
    """Map a raw heading string to one of the canonical research paper sections."""
    cleaned = re.sub(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+", "", heading_text.strip()).lower()
    cleaned = re.sub(r"[:.\-]+$", "", cleaned).strip()
    if not cleaned:
        return None
    for canonical, keywords in CANONICAL_RULES:
        for kw in keywords:
            if cleaned == kw or cleaned.startswith(f"{kw} ") or cleaned.endswith(f" {kw}"):
                return canonical
    return None


def parse_sections(pages: list[PageContent]) -> list[SectionBlock]:
    """
    Parse page-level text into contiguous canonical SectionBlocks preserving start/end pages.
    """
    if not pages:
        return []

    events: list[tuple[int, str, str, list[str]]] = []
    current_canonical = "Front Matter"
    current_raw = "Title & Metadata"
    current_start_page = pages[0].page_number
    buffer: list[str] = []

    for page in pages:
        lines = page.text.splitlines()
        for line in lines:
            stripped = line.strip()
            matched_canonical: str | None = None
            if 3 <= len(stripped) <= 75 and not stripped.endswith("."):
                m = LINE_HEADING_RE.match(stripped)
                if m:
                    matched_canonical = classify_heading(stripped)
            elif stripped.lower().startswith("abstract"):
                matched_canonical = "Abstract"

            if matched_canonical and matched_canonical != current_canonical:
                if buffer:
                    events.append((current_start_page, current_canonical, current_raw, buffer))
                current_canonical = matched_canonical
                current_raw = stripped
                current_start_page = page.page_number
                buffer = []
                # If inline "Abstract—..." keep remainder
                if matched_canonical == "Abstract" and len(stripped) > 12:
                    remainder = re.sub(r"^abstract[:.\s—\-]*", "", stripped, flags=re.IGNORECASE).strip()
                    if remainder:
                        buffer.append(remainder)
            else:
                buffer.append(line)

    if buffer:
        events.append((current_start_page, current_canonical, current_raw, buffer))

    # Merge or build SectionBlocks with accurate end_page
    blocks: list[SectionBlock] = []
    for idx, (start_p, canonical, raw_h, lines_buf) in enumerate(events):
        if canonical == "Front Matter" and len("\n".join(lines_buf).strip()) < 40:
            continue
        end_p = events[idx + 1][0] if idx + 1 < len(events) else pages[-1].page_number
        end_p = max(start_p, end_p)
        content = "\n".join(lines_buf).strip()
        if not content:
            continue
        summary = _extractive_summary(content, max_sentences=3)
        blocks.append(
            SectionBlock(
                canonical_name=canonical,
                raw_heading=raw_h,
                start_page=start_p,
                end_page=end_p,
                content=content,
                summary=summary,
            )
        )

    # Fallback if no canonical sections detected
    if not blocks:
        full_text = "\n\n".join(p.text for p in pages).strip()
        blocks.append(
            SectionBlock(
                canonical_name="Full Document",
                raw_heading="Extracted Text",
                start_page=pages[0].page_number,
                end_page=pages[-1].page_number,
                content=full_text,
                summary=_extractive_summary(full_text, max_sentences=4),
            )
        )
    return blocks


def _extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    picked = [s for s in sentences if len(s) > 25][:max_sentences]
    return " ".join(picked)
