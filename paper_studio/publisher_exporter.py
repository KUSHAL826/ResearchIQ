"""Studio Stage 12: IEEE / Springer / ACM Formatting, Journal Discovery & DOCX/PDF/LaTeX/BibTeX Export."""

from __future__ import annotations

import io
import re
from typing import Any

import docx
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
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


ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def _set_two_column_section(section, num_cols: int = 2, space_twips: int = 360) -> None:
    """Configure an OOXML section for multi-column IEEE layout."""
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), str(space_twips))


def _set_cell_border(cell, **kwargs) -> None:
    """Set explicit borders on a docx table cell (used for clean IEEE top/bottom rules)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right"):
        edge_data = kwargs.get(edge)
        if edge_data is not None:
            tag = f"w:{edge}"
            element = tc_borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tc_borders.append(element)
            for key, val in edge_data.items():
                attr_name = key if ":" in key else f"w:{key}"
                element.set(qn(attr_name), str(val))


def _remove_table_borders(table) -> None:
    """Make author layout grid borderless."""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tbl_pr.append(borders)


def _add_ieee_run_in(doc: docx.Document, head_label: str, body_text: str) -> None:
    """Add IEEE-style run-in bold-italic head (e.g., Abstract— / Index Terms—)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.05

    r_head = p.add_run(f"{head_label}—")
    r_head.font.name = "Times New Roman"
    r_head.font.size = Pt(9)
    r_head.bold = True
    r_head.italic = True

    r_body = p.add_run(body_text.strip())
    r_body.font.name = "Times New Roman"
    r_body.font.size = Pt(9)
    r_body.bold = True


def _add_ieee_results_table(doc: docx.Document, project: ResearchProject) -> None:
    """Insert a real IEEE-styled TABLE I when verified experimental metrics exist."""
    er = project.experiment_results
    headers: list[str] = []
    rows_data: list[list[str]] = []

    if er.computed_classification_metrics:
        m = er.computed_classification_metrics
        headers = ["Evaluation Metric", "Value", "Sample Size"]
        rows_data = [
            ["Accuracy", str(m.get("accuracy", "-")), str(m.get("samples", "-"))],
            ["Precision (Macro)", str(m.get("precision_macro", "-")), str(m.get("samples", "-"))],
            ["Recall (Macro)", str(m.get("recall_macro", "-")), str(m.get("samples", "-"))],
            ["F1-Score (Macro)", str(m.get("f1_macro", "-")), str(m.get("samples", "-"))],
        ]
    elif er.datasets and er.datasets[0].numeric_summary:
        ds = er.datasets[0]
        headers = ["Metric / Column", "Mean", "Std Dev", "Range"]
        for col, stats in list(ds.numeric_summary.items())[:8]:
            rows_data.append(
                [
                    col,
                    str(stats.get("mean", "-")),
                    str(stats.get("std", "-")),
                    f"{stats.get('min', '-')} – {stats.get('max', '-')}",
                ]
            )

    if not headers or not rows_data:
        return

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(3)
    rcap = cap.add_run("TABLE I\n")
    rcap.font.name = "Times New Roman"
    rcap.font.size = Pt(8)
    rcap2 = cap.add_run("SUMMARY OF VERIFIED EXPERIMENTAL RESULTS")
    rcap2.font.name = "Times New Roman"
    rcap2.font.size = Pt(8)

    table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
    table.alignment = 1
    _remove_table_borders(table)

    for c_idx, h_txt in enumerate(headers):
        cell = table.cell(0, c_idx)
        _set_cell_border(
            cell,
            top={"val": "single", "sz": "8", "color": "000000"},
            bottom={"val": "single", "sz": "6", "color": "000000"},
        )
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_txt)
        run.font.name = "Times New Roman"
        run.font.size = Pt(8)
        run.bold = True

    for r_idx, row_vals in enumerate(rows_data, start=1):
        is_last = r_idx == len(rows_data)
        for c_idx, val in enumerate(row_vals):
            cell = table.cell(r_idx, c_idx)
            if is_last:
                _set_cell_border(cell, bottom={"val": "single", "sz": "8", "color": "000000"})
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(val)
            run.font.name = "Times New Roman"
            run.font.size = Pt(8)


def _export_ieee_conference_docx(project: ResearchProject) -> bytes:
    """
    Draft Microsoft Word (.docx) strictly following IEEEtran conference template
    (IEEE-conference-template-062824):
    - Letter page, 0.75" top, 1.0" bottom, 0.625" left/right margins
    - Full-width centered 24pt title & multi-column author/affiliation block
    - Continuous 2-column body with 0.25" gutter
    - Run-in bold-italic Abstract— and Index Terms—
    - Centered Roman-numeral uppercase section heads (I. INTRODUCTION)
    - Italic lettered subheadings (A. Subheading)
    - IEEE-ruled TABLE I for verified experimental metrics
    - Numbered bracketed 8pt hanging-indent REFERENCES
    """
    doc = docx.Document()

    # Section 0: Single-column header for Title & Author Block
    sec0 = doc.sections[0]
    sec0.page_width = Inches(8.5)
    sec0.page_height = Inches(11.0)
    sec0.top_margin = Inches(0.75)
    sec0.bottom_margin = Inches(1.0)
    sec0.left_margin = Inches(0.625)
    sec0.right_margin = Inches(0.625)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    title = (project.proposal.title or project.selected_topic or project.title).strip()
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(10)
    r_title = p_title.add_run(title)
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(24)

    # Author grid (up to 3 authors per row, matching IEEEtran \IEEEauthorblockN / \IEEEauthorblockA)
    authors = project.publication_prep.authors or ["1st Given Name Surname"]
    affil = project.publication_prep.affiliation or "Department / Organization"
    cols_per_row = min(3, max(1, len(authors)))
    num_rows = (len(authors) + cols_per_row - 1) // cols_per_row
    auth_table = doc.add_table(rows=num_rows, cols=cols_per_row)
    auth_table.alignment = 1
    _remove_table_borders(auth_table)

    for idx, author_name in enumerate(authors):
        r_i = idx // cols_per_row
        c_i = idx % cols_per_row
        cell = auth_table.cell(r_i, c_i)
        p_a = cell.paragraphs[0]
        p_a.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_a.paragraph_format.space_after = Pt(8)
        rn = p_a.add_run(f"{author_name}\n")
        rn.font.name = "Times New Roman"
        rn.font.size = Pt(11)
        ra = p_a.add_run(f"{affil}\n")
        ra.font.name = "Times New Roman"
        ra.font.size = Pt(10)
        ra.italic = True

    # Section 1: Continuous Two-Column IEEE Body
    sec_body = doc.add_section(WD_SECTION.CONTINUOUS)
    sec_body.top_margin = Inches(0.75)
    sec_body.bottom_margin = Inches(1.0)
    sec_body.left_margin = Inches(0.625)
    sec_body.right_margin = Inches(0.625)
    _set_two_column_section(sec_body, num_cols=2, space_twips=360)

    sections = _collect_sections_preserving_approved(project)
    sec_dict = dict(sections)

    abstract_body = sec_dict.get("Abstract") or f"This paper investigates {title} in the domain of {project.domain}."
    _add_ieee_run_in(doc, "Abstract", abstract_body)

    keywords_list = project.publication_prep.keywords or [project.domain.lower(), "research intelligence", "empirical evaluation"]
    _add_ieee_run_in(doc, "Index Terms", ", ".join(k.lower() for k in keywords_list) + ".")

    roman_idx = 0
    sub_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    subhead_candidates = {
        "Motivation", "Problem Statement", "Scope", "Research Objectives", "Research Questions",
        "Hypotheses", "Research Design", "Proposed Architecture", "Data Flow",
        "Mathematical Formulation", "Algorithms & Components", "Technology Stack",
        "Preprocessing Steps", "Module Organization", "Reproducibility Notes",
        "Baselines", "Evaluation Metrics", "Configurations", "Ablation Plan", "Discussion",
    }

    for sec_name, content in sections:
        if sec_name == "Abstract":
            continue

        if sec_name == "References":
            p_ref_h = doc.add_paragraph()
            p_ref_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_ref_h.paragraph_format.space_before = Pt(10)
            p_ref_h.paragraph_format.space_after = Pt(4)
            rr = p_ref_h.add_run("REFERENCES")
            rr.font.name = "Times New Roman"
            rr.font.size = Pt(10)

            for line in [ln.strip() for ln in content.splitlines() if ln.strip()]:
                p_ref = doc.add_paragraph()
                p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_ref.paragraph_format.left_indent = Inches(0.22)
                p_ref.paragraph_format.first_line_indent = Inches(-0.22)
                p_ref.paragraph_format.space_after = Pt(2)
                r_txt = p_ref.add_run(line)
                r_txt.font.name = "Times New Roman"
                r_txt.font.size = Pt(8)
            continue

        # Numbered Roman-Numeral Centered Section Heading
        r_num = ROMAN_NUMERALS[roman_idx] if roman_idx < len(ROMAN_NUMERALS) else str(roman_idx + 1)
        roman_idx += 1
        p_h = doc.add_paragraph()
        p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_h.paragraph_format.space_before = Pt(10)
        p_h.paragraph_format.space_after = Pt(4)
        p_h.paragraph_format.keep_with_next = True
        rh = p_h.add_run(f"{r_num}. {sec_name.upper()}")
        rh.font.name = "Times New Roman"
        rh.font.size = Pt(10)

        sub_idx = 0
        first_para = True
        for block in content.split("\n\n"):
            cleaned = re.sub(r"^#+\s*", "", block.strip())
            if not cleaned:
                continue

            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            # Detect subheading line followed by body lines
            if len(lines) >= 2 and (lines[0] in subhead_candidates or (len(lines[0]) < 42 and not lines[0].endswith((".", ":", ",", ";")))):
                sub_title = lines[0]
                letter = sub_letters[sub_idx % len(sub_letters)]
                sub_idx += 1
                p_sub = doc.add_paragraph()
                p_sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p_sub.paragraph_format.space_before = Pt(5)
                p_sub.paragraph_format.space_after = Pt(2)
                p_sub.paragraph_format.keep_with_next = True
                rs = p_sub.add_run(f"{letter}. {sub_title}")
                rs.font.name = "Times New Roman"
                rs.font.size = Pt(10)
                rs.italic = True
                body_lines = lines[1:]
                first_para = True
                for bl in body_lines:
                    _add_ieee_body_paragraph(doc, bl, indent=not first_para and not bl.startswith(("-", "*", "•")))
                    first_para = False
            else:
                for bl in lines:
                    _add_ieee_body_paragraph(doc, bl, indent=not first_para and not bl.startswith(("-", "*", "•")))
                    first_para = False

        if sec_name == "Results and Discussion":
            _add_ieee_results_table(doc, project)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _add_ieee_body_paragraph(doc: docx.Document, text: str, indent: bool = True) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.05
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.14)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)


def export_docx_bytes(project: ResearchProject) -> bytes:
    """Export the manuscript to a formatted Microsoft Word (.docx) binary matching the target conference style."""
    if project.publication_prep.target_format == "IEEE":
        return _export_ieee_conference_docx(project)

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
