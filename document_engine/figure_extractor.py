"""Figure and equation extraction from PDF pages."""

from __future__ import annotations

import re

from core.schemas import ExtractedFigure, PageContent

FIGURE_CAPTION_RE = re.compile(r"((?:Fig\.?|Figure)\s*\d+[:.\-\s][^\n]{5,180})", re.IGNORECASE)
EQUATION_RE = re.compile(r"([^\n]{5,120}(?:=|\u2211|\u222b|\u2208|\u2264|\u2265)[^\n]{3,120}\(\d+\))")


def extract_figures_and_equations(pages: list[PageContent]) -> list[ExtractedFigure]:
    """Identify figure captions and numbered equations with page numbers and surrounding context."""
    items: list[ExtractedFigure] = []
    seen_captions: set[str] = set()

    for page in pages:
        for match in FIGURE_CAPTION_RE.finditer(page.text):
            cap = re.sub(r"\s+", " ", match.group(1)).strip()
            key = f"fig_{page.page_number}_{cap[:40].lower()}"
            if key in seen_captions:
                continue
            seen_captions.add(key)
            start = max(0, match.start() - 120)
            end = min(len(page.text), match.end() + 180)
            items.append(
                ExtractedFigure(
                    page_number=page.page_number,
                    kind="figure",
                    caption=cap,
                    context_snippet=re.sub(r"\s+", " ", page.text[start:end]).strip(),
                )
            )

        for match in EQUATION_RE.finditer(page.text):
            eq_text = re.sub(r"\s+", " ", match.group(1)).strip()
            key = f"eq_{page.page_number}_{eq_text[:40].lower()}"
            if key in seen_captions:
                continue
            seen_captions.add(key)
            items.append(
                ExtractedFigure(
                    page_number=page.page_number,
                    kind="equation",
                    caption=f"Equation (p.{page.page_number})",
                    context_snippet=eq_text,
                )
            )

    return items[:60]
