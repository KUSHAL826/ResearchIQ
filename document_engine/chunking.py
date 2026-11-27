"""Section-aware chunking, hierarchical summarization, and end-to-end PDF pipeline."""

from __future__ import annotations

from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.schemas import DocumentChunk, PageContent, ParsedPaper, SectionBlock
from document_engine.figure_extractor import extract_figures_and_equations
from document_engine.pdf_extractor import extract_pdf_pages
from document_engine.section_parser import parse_sections
from document_engine.table_extractor import extract_tables_from_pdf


def create_section_aware_chunks(
    paper_id: str,
    paper_title: str,
    sections: list[SectionBlock],
    pages: list[PageContent],
    chunk_size: int = 1100,
    chunk_overlap: int = 180,
) -> list[DocumentChunk]:
    """Split each section into overlapping chunks tagged with canonical section and page range."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []

    for sec in sections:
        pieces = splitter.split_text(sec.content)
        for piece in pieces:
            clean_piece = piece.strip()
            if len(clean_piece) < 30:
                continue
            # Estimate exact page inside section span
            matched_pages = [
                p.page_number
                for p in pages
                if sec.start_page <= p.page_number <= sec.end_page and clean_piece[:60] in p.text
            ]
            p_start = matched_pages[0] if matched_pages else sec.start_page
            p_end = matched_pages[-1] if matched_pages else sec.end_page
            chunks.append(
                DocumentChunk(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    section=sec.canonical_name,
                    page_start=p_start,
                    page_end=p_end,
                    text=clean_piece,
                    token_estimate=max(1, len(clean_piece) // 4),
                )
            )
    return chunks


def build_hierarchical_summary(title: str, sections: list[SectionBlock]) -> str:
    """Create a structured hierarchical summary from section summaries."""
    parts = [f"Document: {title}"]
    for sec in sections:
        if sec.summary:
            parts.append(f"- {sec.canonical_name} (pp. {sec.start_page}-{sec.end_page}): {sec.summary}")
    return "\n".join(parts)


def process_pdf_bytes(
    raw_bytes: bytes,
    filename: str,
    progress_cb: Callable[[str], None] | None = None,
) -> ParsedPaper:
    """
    Execute the real document processing pipeline:
    1. Document uploaded -> 2. Text extracted -> 3. Sections detected -> Chunks built.
    """
    completed: list[str] = ["Document uploaded"]
    if progress_cb:
        progress_cb("Document uploaded")

    pages, meta = extract_pdf_pages(raw_bytes, filename=filename)
    completed.append("Text extracted")
    if progress_cb:
        progress_cb("Text extracted")

    sections = parse_sections(pages)
    tables = extract_tables_from_pdf(raw_bytes)
    figures = extract_figures_and_equations(pages)
    completed.append("Sections detected")
    if progress_cb:
        progress_cb("Sections detected")

    paper = ParsedPaper(
        filename=filename,
        title=meta["title"],
        authors=meta["authors"],
        year=meta["year"],
        doi=meta["doi"],
        num_pages=meta["num_pages"],
        file_hash=meta["file_hash"],
        pages=pages,
        sections=sections,
        tables=tables,
        figures=figures,
        hierarchical_summary=build_hierarchical_summary(meta["title"], sections),
        processing_stages_completed=completed,
    )
    paper.chunks = create_section_aware_chunks(
        paper_id=paper.paper_id,
        paper_title=paper.title,
        sections=sections,
        pages=pages,
    )
    return paper
