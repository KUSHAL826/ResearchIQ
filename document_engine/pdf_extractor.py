"""PDF validation and page-level extraction using PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import re
from typing import Tuple

import pymupdf as fitz  # PyMuPDF

from core.config import MAX_UPLOAD_PDF_BYTES
from core.schemas import PageContent

DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
YEAR_PATTERN = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
HEADING_CANDIDATE_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+)?([A-Z][A-Za-z0-9\s,&:\-/]{2,70})$"
)


def validate_pdf_bytes(raw_bytes: bytes, filename: str = "paper.pdf") -> None:
    """Validate uploaded PDF bytes for non-emptiness, size limit, magic header, and openability."""
    if not raw_bytes:
        raise ValueError(f"'{filename}' is empty.")
    if len(raw_bytes) > MAX_UPLOAD_PDF_BYTES:
        raise ValueError(f"'{filename}' exceeds {MAX_UPLOAD_PDF_BYTES // (1024 * 1024)} MB upload limit.")
    if not raw_bytes[:10].lstrip().startswith(b"%PDF"):
        raise ValueError(f"'{filename}' does not appear to be a valid PDF file.")
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        if doc.page_count == 0:
            doc.close()
            raise ValueError(f"'{filename}' contains zero pages.")
        doc.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{filename}': {exc}") from exc


def _detect_page_headings(text: str) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > 85 or len(line) < 4:
            continue
        if line.endswith(".") and not line.lower().startswith("abstract"):
            continue
        m = HEADING_CANDIDATE_RE.match(line)
        if m:
            candidate = m.group(1).strip()
            if len(candidate.split()) <= 9:
                headings.append(line)
    return headings[:15]


def extract_pdf_pages(raw_bytes: bytes, filename: str = "paper.pdf") -> Tuple[list[PageContent], dict]:
    """
    Extract page-level text and PDF metadata from raw PDF bytes.
    Returns (pages, metadata_dict).
    """
    validate_pdf_bytes(raw_bytes, filename=filename)
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    pages: list[PageContent] = []
    try:
        raw_meta = doc.metadata or {}
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            text = page.get_text("text") or ""
            # Clean excessive blank lines while preserving paragraph breaks
            cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages.append(
                PageContent(
                    page_number=idx + 1,
                    text=cleaned,
                    char_count=len(cleaned),
                    detected_headings=_detect_page_headings(cleaned),
                )
            )
    finally:
        doc.close()

    first_page_text = pages[0].text if pages else ""
    title = (raw_meta.get("title") or "").strip()
    if not title or title.lower() in {"untitled", "microsoft word", "latex"} or len(title) < 5:
        # Infer title from first substantial lines of page 1
        lines = [ln.strip() for ln in first_page_text.splitlines() if len(ln.strip()) > 8]
        title = lines[0][:180] if lines else filename.rsplit(".", 1)[0]

    authors_raw = (raw_meta.get("author") or "").strip()
    authors = [a.strip() for a in re.split(r"[,;]| and ", authors_raw) if a.strip()] if authors_raw else []

    doi_match = DOI_PATTERN.search("\n".join(p.text for p in pages[:2]))
    doi = doi_match.group(1).rstrip(".,)") if doi_match else ""

    year: int | None = None
    creation_date = raw_meta.get("creationDate") or ""
    date_match = re.search(r"(19\d\d|20\d\d)", creation_date)
    if date_match:
        year = int(date_match.group(1))
    else:
        y_match = YEAR_PATTERN.search(first_page_text[:1500])
        if y_match:
            year = int(y_match.group(1))

    metadata = {
        "title": title,
        "authors": authors,
        "doi": doi,
        "year": year,
        "num_pages": len(pages),
        "file_hash": file_hash,
    }
    return pages, metadata
