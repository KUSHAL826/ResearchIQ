"""Studio Stage 12: IEEE / Springer / ACM Formatting, Journal Discovery & DOCX/PDF/LaTeX/BibTeX Export."""

from __future__ import annotations

import io
import re
from typing import Any

import docx
from docx.shared import Pt
import requests
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from agents.paper_writer import build_reference_list
from core.config import MANUSCRIPT_SECTIONS, OPENALEX_BASE_URL
from core.schemas import ResearchProject
from paper_studio.manuscript_manager import compute_manuscript_health


FORMAT_GUIDELINES = {
    "IEEE": [
        {"item": "Two-column IEEEtran conference/journal structure", "rule": "Title, Author Block, Abstract (150-250 words), Index Terms, Sections I-VI"},
        {"item": "Numbered bracket citations [1], [2]", "rule": "Ordered sequentially by appearance"},
        {"item": "Active voice & quantitative table captions above tables", "rule": "Figure captions below figures"},
    ],
    "Springer": [
        {"item": "LNCS / Springer Nature single-column layout", "rule": "Title, Authors, ORCID/Affiliation, Abstract, Keywords, Numbered Sections"},
        {"item": "Numbered references or Author-Year depending on series", "rule": "Include DOI links when available"},
        {"item": "Clear ethical / data availability statement", "rule": "Explicit dataset & reproducibility details"},
    ],
    "ACM": [
        {"item": "ACM acmart double/single-column format", "rule": "CCS Concepts, Keywords, ACM Reference Format block"},
        {"item": "Structured Evaluation & Limitations discussion", "rule": "Threats to validity explicitly addressed"},
        {"item": "Artifact & reproducibility checklist", "rule": "Document code/dataset availability"},
    ],
}


def discover_target_journals(topic_or_domain: str, limit: int = 8) -> tuple[list[dict[str, Any]], str | None]:
    """Query OpenAlex /sources API for real journals/venues matching the research topic."""
    try:
        resp = requests.get(
            f"{OPENALEX_BASE_URL}/sources",
            params={"search": topic_or_domain, "per-page": limit},
            timeout=12,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return [], f"Journal lookup failed: {exc}"

    journals: list[dict[str, Any]] = []
    for src in payload.get("results", []):
        journals.append(
            {
                "name": src.get("display_name", "Unknown Venue"),
                "type": src.get("type", "journal"),
                "issn": ", ".join(src.get("issn") or []) or "N/A",
                "is_oa": "Open Access" if src.get("is_oa") else "Subscription / Hybrid",
                "works_count": src.get("works_count"),
                "cited_by_count": src.get("cited_by_count"),
                "homepage": src.get("homepage_url") or src.get("id") or "",
            }
        )
    return journals, None


def assess_paper_readiness(project: ResearchProject) -> dict[str, Any]:
    """Evaluate readiness score from manuscript completion, approval, consistency, and citations."""
    health = compute_manuscript_health(project)
    comp = health["completion"]
    consistency_score = health["consistency"].overall_score
    ref_ok = health["reference_check"]["is_complete"]

    readiness = (
        comp["completion_pct"] * 0.45
        + consistency_score * 0.40
        + (15.0 if ref_ok else 5.0)
    )
    readiness = round(min(100.0, max(0.0, readiness)), 1)
    return {
        "readiness_score": readiness,
        "drafted_sections": f"{comp['drafted_count']}/{comp['total_sections']}",
        "approved_sections": f"{comp['approved_count']}/{comp['total_sections']}",
        "total_words": comp["total_words"],
        "consistency_score": consistency_score,
        "references_ready": ref_ok,
        "guidelines": FORMAT_GUIDELINES.get(project.publication_prep.target_format, FORMAT_GUIDELINES["IEEE"]),
    }


def _collect_sections_preserving_approved(project: ResearchProject) -> list[tuple[str, str]]:
    """Collect manuscript sections in canonical order, preserving exact approved content."""
    ordered: list[tuple[str, str]] = []
    for sec_name in MANUSCRIPT_SECTIONS:
        sec = project.manuscript_sections.get(sec_name)
        text = sec.content.strip() if sec and sec.content.strip() else ""
        if sec_name == "References" and not text:
            refs = build_reference_list(project, style=project.publication_prep.target_format)
            text = "\n\n".join(refs)
        if text:
            ordered.append((sec_name, text))
    return ordered


def export_bibtex(project: ResearchProject) -> str:
    """Generate a valid .bib file from real session PDFs and discovered works."""
    entries: list[str] = []
    seen_keys: set[str] = set()

    def _make_key(author_str: str, year_val: Any, title_str: str) -> str:
        first_author = re.sub(r"[^A-Za-z]", "", (author_str.split(",")[0].split()[-1] if author_str else "Author")) or "Author"
        yr = str(year_val or "nd")
        word = re.sub(r"[^A-Za-z]", "", (title_str.split()[0] if title_str else "paper")).lower()
        key = f"{first_author}{yr}{word}"
        idx = 1
        base = key
        while key in seen_keys:
            idx += 1
            key = f"{base}{idx}"
        seen_keys.add(key)
        return key

    for p in project.parsed_papers:
        authors_str = " and ".join(p.authors) if p.authors else "Unknown Author"
        key = _make_key(authors_str, p.year, p.title)
        doi_line = f"  doi = {{{p.doi}}},\n" if p.doi else ""
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{p.title}}},\n"
            f"  author = {{{authors_str}}},\n"
            f"  year = {{{p.year or '2024'}}},\n"
            f"{doi_line}"
            f"}}"
        )

    included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
    pool = included if included else [w for w in project.discovered_works if not w.is_duplicate][:15]
    for w in pool:
        authors_str = " and ".join(w.authors) if w.authors else "Unknown Author"
        key = _make_key(authors_str, w.year, w.title)
        doi_line = f"  doi = {{{w.doi}}},\n" if w.doi else ""
        url_line = f"  url = {{{w.url}}},\n" if w.url else ""
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{w.title}}},\n"
            f"  author = {{{authors_str}}},\n"
            f"  journal = {{{w.venue or w.source}}},\n"
            f"  year = {{{w.year or '2024'}}},\n"
            f"{doi_line}{url_line}"
            f"}}"
        )
    return "\n\n".join(entries) if entries else "% No references added to session yet."


def export_latex(project: ResearchProject) -> str:
    """Export the manuscript as a clean LaTeX document tailored to IEEE, Springer, or ACM."""
    fmt = project.publication_prep.target_format
    title = project.proposal.title or project.selected_topic or project.title
    authors = ", ".join(project.publication_prep.authors)
    affiliation = project.publication_prep.affiliation
    keywords = ", ".join(project.publication_prep.keywords or [project.domain])

    if fmt == "IEEE":
        doc_class = "\\documentclass[conference]{IEEEtran}"
    elif fmt == "Springer":
        doc_class = "\\documentclass[runningheads]{llncs}"
    else:
        doc_class = "\\documentclass[sigconf]{acmart}"

    sections = _collect_sections_preserving_approved(project)
    body_parts: list[str] = []
    abstract_text = ""
    for sec_name, content in sections:
        clean = content.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
        if sec_name == "Abstract":
            abstract_text = clean
        elif sec_name == "References":
            ref_lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
            items = "\n".join(f"\\bibitem{{ref{i+1}}} {ln}" for i, ln in enumerate(ref_lines))
            body_parts.append(f"\\begin{{thebibliography}}{{99}}\n{items}\n\\end{{thebibliography}}")
        else:
            body_parts.append(f"\\section{{{sec_name}}}\n{clean}")

    return (
        f"{doc_class}\n"
        f"\\usepackage{{amsmath,amssymb,graphicx,hyperref}}\n"
        f"\\begin{{document}}\n"
        f"\\title{{{title}}}\n"
        f"\\author{{{authors} \\\\ {affiliation}}}\n"
        f"\\maketitle\n\n"
        f"\\begin{{abstract}}\n{abstract_text or 'Abstract pending.'}\n\\end{{abstract}}\n\n"
        f"\\textbf{{Keywords:}} {keywords}\n\n"
        + "\n\n".join(body_parts)
        + "\n\n\\end{document}\n"
    )


def export_docx_bytes(project: ResearchProject) -> bytes:
    """Export the manuscript to a formatted Microsoft Word (.docx) binary."""
    doc = docx.Document()
    title = project.proposal.title or project.selected_topic or project.title
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    h = doc.add_heading(title, level=0)
    h.alignment = 1
    meta_p = doc.add_paragraph(
        f"{', '.join(project.publication_prep.authors)} | {project.publication_prep.affiliation}\n"
        f"Target Style: {project.publication_prep.target_format}"
    )
    meta_p.alignment = 1

    for sec_name, content in _collect_sections_preserving_approved(project):
        doc.add_heading(sec_name, level=1)
        for para in content.split("\n\n"):
            cleaned = re.sub(r"^#+\s*", "", para.strip())
            if cleaned:
                doc.add_paragraph(cleaned)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_pdf_bytes(project: ResearchProject) -> bytes:
    """Export the manuscript to a clean academic PDF using ReportLab."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("RIQTitle", parent=styles["Title"], fontSize=17, leading=21, spaceAfter=8)
    meta_style = ParagraphStyle("RIQMeta", parent=styles["Normal"], fontSize=9.5, leading=13, spaceAfter=14)
    sec_style = ParagraphStyle("RIQSec", parent=styles["Heading2"], fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("RIQBody", parent=styles["Normal"], fontSize=10.5, leading=14.5, spaceAfter=6)

    title = project.proposal.title or project.selected_topic or project.title
    story = [
        Paragraph(_xml_escape(title), title_style),
        Paragraph(
            _xml_escape(
                f"{', '.join(project.publication_prep.authors)} — {project.publication_prep.affiliation} "
                f"[{project.publication_prep.target_format} Draft]"
            ),
            meta_style,
        ),
    ]

    sections = _collect_sections_preserving_approved(project)
    if not sections:
        story.append(Paragraph("No manuscript sections have been drafted yet.", body_style))
    else:
        for sec_name, content in sections:
            story.append(Paragraph(_xml_escape(sec_name), sec_style))
            for para in content.split("\n\n"):
                cleaned = re.sub(r"^#+\s*", "", para.strip())
                if cleaned:
                    story.append(Paragraph(_xml_escape(cleaned).replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
