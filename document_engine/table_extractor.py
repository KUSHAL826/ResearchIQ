"""Table extraction from PDFs using pdfplumber."""

from __future__ import annotations

import io
import re

import pdfplumber

from core.schemas import ExtractedTable


def extract_tables_from_pdf(raw_bytes: bytes, max_tables: int = 30) -> list[ExtractedTable]:
    """Extract structured tables with page numbers and nearby caption hints."""
    tables_out: list[ExtractedTable] = []
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                caption_matches = re.findall(r"(Table\s+\d+[:.\-\s][^\n]{5,120})", page_text, flags=re.IGNORECASE)
                raw_tables = page.extract_tables() or []
                for t_idx, raw_table in enumerate(raw_tables):
                    cleaned_rows: list[list[str]] = []
                    for row in raw_table:
                        if not row:
                            continue
                        cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
                        if any(cells):
                            cleaned_rows.append(cells)
                    if len(cleaned_rows) < 2:
                        continue
                    headers = cleaned_rows[0]
                    body_rows = cleaned_rows[1:25]
                    caption = caption_matches[t_idx] if t_idx < len(caption_matches) else f"Table on Page {page_idx}"
                    tables_out.append(
                        ExtractedTable(
                            page_number=page_idx,
                            caption=caption.strip(),
                            headers=headers,
                            rows=body_rows,
                        )
                    )
                    if len(tables_out) >= max_tables:
                        return tables_out
    except Exception:
        return tables_out
    return tables_out
