"""Studio Stage 12: IEEE / Springer / ACM Draft Formatting, Journal Discovery & DOCX/PDF/LaTeX/BibTeX Export."""

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

ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

FORMAT_GUIDELINES = {
    "IEEE": [
        {"item": "Draft Two-Column IEEEtran Conference Layout", "rule": "Title, Author Block, run-in Abstract— & Index Terms—, Roman-numeral sections I–X (verify with official IEEEtran before submission)"},
        {"item": "Numbered bracket citations [1], [2]", "rule": "Ordered sequentially by appearance"},
        {"item": "Quantitative TABLE I captions above tables", "rule": "Figure captions below figures"},
    ],
    "Springer": [
        {"item": "Draft LNCS-Style Single-Column Layout", "rule": "Title, Authors/Affiliation, Abstract, Keywords, Numbered Sections 1–10 (compile with official llncs.cls for final submission)"},
        {"item": "Numbered references with DOI links", "rule": "Consistent bibliographic ordering"},
        {"item": "Ethical / data availability & reproducibility details", "rule": "Explicit dataset & split reporting"},
    ],
    "ACM": [
        {"item": "Draft ACM-Style Conference Layout", "rule": "Title, Authors/Affiliation, Abstract, CCS Concepts/Keywords, Numbered Sections (compile with official acmart.cls for final submission)"},
        {"item": "Structured Evaluation & Threats to Validity", "rule": "Limitations & baselines clearly documented"},
        {"item": "Artifact & reproducibility statement", "rule": "Document code/dataset availability"},
    ],
}

TEMPLATE_COMPLIANCE_NOTE = (
    "Note: Exported DOCX, PDF, and LaTeX files follow structural draft conventions for the selected venue style "
    "(IEEE/Springer/ACM). Compile with the venue's official LaTeX .cls or Word .dotx template before final submission."
)


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
        "compliance_note": TEMPLATE_COMPLIANCE_NOTE,
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


def _escape_latex_preserving_math(text: str) -> str:
    """Escape LaTeX special characters in prose while preserving $...$ and $$...$$ math blocks."""
    math_pattern = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)
    parts = math_pattern.split(text)
    out: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            # Math span — keep verbatim
            out.append(part)
        else:
            escaped = (
                part.replace("\\", "\\textbackslash{}")
                .replace("&", "\\&")
                .replace("%", "\\%")
                .replace("#", "\\#")
                .replace("_", "\\_")
                .replace("~", "\\textasciitilde{}")
                .replace("^", "\\textasciicircum{}")
            )
            out.append(escaped)
    return "".join(out)


def _build_latex_results_table(project: ResearchProject) -> str:
    """Generate a valid LaTeX Table I from verified Stage 9 metrics if present."""
    er = project.experiment_results
    rows: list[list[str]] = []
    headers: list[str] = []
    if er.computed_classification_metrics:
        m = er.computed_classification_metrics
        headers = ["Metric", "Value", "Samples"]
        rows = [
            ["Accuracy", str(m.get("accuracy", "-")), str(m.get("samples", "-"))],
            ["Precision (Macro)", str(m.get("precision_macro", "-")), str(m.get("samples", "-"))],
            ["Recall (Macro)", str(m.get("recall_macro", "-")), str(m.get("samples", "-"))],
            ["F1-Score (Macro)", str(m.get("f1_macro", "-")), str(m.get("samples", "-"))],
        ]
    elif er.datasets and er.datasets[0].numeric_summary:
        headers = ["Column", "Mean", "Std", "Range"]
        for col, stats in list(er.datasets[0].numeric_summary.items())[:8]:
            rows.append([col, str(stats.get("mean", "-")), str(stats.get("std", "-")), f"{stats.get('min', '-')}--{stats.get('max', '-')}"])
    if not headers or not rows:
        return ""

    col_spec = "|l|" + "c|" * (len(headers) - 1)
    header_line = " & ".join(f"\\textbf{{{_escape_latex_preserving_math(h)}}}" for h in headers) + " \\\\ \\hline"
    row_lines = "\n".join(
        " & ".join(_escape_latex_preserving_math(v) for v in r) + " \\\\ \\hline"
        for r in rows
    )
    return (
        "\\begin{table}[htbp]\n"
        "\\caption{Summary of Verified Experimental Results}\n"
        "\\begin{center}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"
        f"{header_line}\n{row_lines}\n"
        "\\end{tabular}\n\\label{tab:results}\n"
        "\\end{center}\n\\end{table}"
    )


def export_bibtex(project: ResearchProject) -> str:
    """Generate a valid .bib file from real session PDFs and discovered works."""
    entries: list[str] = []
    seen_keys: set[str] = set()

    def _make_key(author_str: str, year_val: Any, title_str: str) -> str:
        first_author = re.sub(r"[^A-Za-z]", "", (author_str.split(",")[0].split()[-1] if author_str else "Author")) or "Author"
        yr = str(year_val or "nd")
        word = re.sub(r"[^A-Za-z]", "", (title_str.split()[0] if title_str else "paper")).lower() or "paper"
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
    """Export the manuscript as a clean, valid LaTeX document tailored to IEEE, Springer, or ACM."""
    fmt = project.publication_prep.target_format
    title = _escape_latex_preserving_math(project.proposal.title or project.selected_topic or project.title)
    authors_list = project.publication_prep.authors or ["First Author"]
    affil = _escape_latex_preserving_math(project.publication_prep.affiliation or "Department")
    keywords = _escape_latex_preserving_math(", ".join(project.publication_prep.keywords or [project.domain]))

    if fmt == "IEEE":
        doc_class = "\\documentclass[conference]{IEEEtran}"
        author_blocks = "\n\\and\n".join(
            f"\\IEEEauthorblockN{{{_escape_latex_preserving_math(a)}}}\n\\IEEEauthorblockA{{\\textit{{{affil}}}}}"
            for a in authors_list
        )
        author_tex = f"\\author{{{author_blocks}}}"
        kw_tex = f"\\begin{{IEEEkeywords}}\n{keywords}\n\\end{{IEEEkeywords}}"
    elif fmt == "Springer":
        doc_class = "\\documentclass[runningheads]{llncs}"
        author_tex = (
            "\\author{" + " \\and ".join(_escape_latex_preserving_math(a) for a in authors_list) + "}\n"
            f"\\institute{{{affil}}}"
        )
        kw_tex = f"\\keywords{{{keywords}}}"
    else:
        doc_class = "\\documentclass[sigconf]{acmart}"
        author_tex = "\n".join(
            f"\\author{{{_escape_latex_preserving_math(a)}}}\n\\affiliation{{\\institution{{{affil}}}}}"
            for a in authors_list
        )
        kw_tex = f"\\keywords{{{keywords}}}"

    sections = _collect_sections_preserving_approved(project)
    body_parts: list[str] = []
    abstract_text = ""
    subhead_candidates = {
        "Motivation", "Problem Statement", "Scope", "Research Objectives", "Research Questions",
        "Hypotheses", "Research Design", "Proposed Architecture", "Data Flow",
        "Mathematical Formulation", "Algorithms & Components", "Technology Stack",
        "Preprocessing Steps", "Module Organization", "Reproducibility Notes",
        "Baselines", "Evaluation Metrics", "Configurations", "Ablation Plan", "Discussion",
    }

    for sec_name, content in sections:
        if sec_name == "Abstract":
            abstract_text = _escape_latex_preserving_math(content)
        elif sec_name == "References":
            ref_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            items = "\n".join(f"\\bibitem{{ref{i+1}}} {_escape_latex_preserving_math(ln)}" for i, ln in enumerate(ref_lines))
            body_parts.append(f"\\begin{{thebibliography}}{{99}}\n{items}\n\\end{{thebibliography}}")
        else:
            sec_blocks: list[str] = [f"\\section{{{_escape_latex_preserving_math(sec_name)}}}"]
            for blk in content.split("\n\n"):
                lines = [ln.strip() for ln in blk.strip().splitlines() if ln.strip()]
                if not lines:
                    continue
                if lines[0] in subhead_candidates and len(lines) >= 2:
                    sec_blocks.append(f"\\subsection{{{_escape_latex_preserving_math(lines[0])}}}")
                    sec_blocks.append(_escape_latex_preserving_math("\n".join(lines[1:])))
                else:
                    sec_blocks.append(_escape_latex_preserving_math("\n".join(lines)))
            if sec_name == "Results and Discussion":
                tab_tex = _build_latex_results_table(project)
                if tab_tex:
                    sec_blocks.append(tab_tex)
            body_parts.append("\n\n".join(sec_blocks))

    preamble_extra = "\\usepackage{amsmath,amssymb,graphicx,hyperref}"
    if fmt == "ACM":
        # acmart loads hyperref/graphicx internally
        preamble_extra = "\\usepackage{amsmath,amssymb}"

    maketitle_order = (
        f"\\title{{{title}}}\n{author_tex}\n"
        + (f"\\begin{{abstract}}\n{abstract_text or 'Abstract pending.'}\n\\end{{abstract}}\n{kw_tex}\n\\maketitle"
           if fmt == "ACM"
           else f"\\maketitle\n\n\\begin{{abstract}}\n{abstract_text or 'Abstract pending.'}\n\\end{{abstract}}\n\n{kw_tex}")
    )

    return (
        f"% Draft formatted for {fmt} — compile with official {fmt} class file for camera-ready submission\n"
        f"{doc_class}\n"
        f"{preamble_extra}\n"
        f"\\begin{{document}}\n"
        f"{maketitle_order}\n\n"
        + "\n\n".join(body_parts)
        + "\n\n\\end{document}\n"
    )


def _set_two_column_section(section, num_cols: int = 2, space_twips: int = 360) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), str(space_twips))


def _set_cell_border(cell, **kwargs) -> None:
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
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tbl_pr.append(borders)


def _add_ieee_run_in(doc: docx.Document, head_label: str, body_text: str) -> None:
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


def _add_results_table_docx(doc: docx.Document, project: ResearchProject, font_size_pt: int = 8) -> None:
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
    rcap = cap.add_run("TABLE I\nSUMMARY OF VERIFIED EXPERIMENTAL RESULTS")
    rcap.font.name = "Times New Roman"
    rcap.font.size = Pt(font_size_pt)

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
        run.font.size = Pt(font_size_pt)
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
            run.font.size = Pt(font_size_pt)


def _export_ieee_conference_docx(project: ResearchProject) -> bytes:
    """Draft Word (.docx) formatted per IEEEtran conference structure."""
    doc = docx.Document()
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

    authors = project.publication_prep.authors or ["1st Given Name Surname"]
    affil = project.publication_prep.affiliation or "Department / Organization"
    cols_per_row = min(3, max(1, len(authors)))
    num_rows = (len(authors) + cols_per_row - 1) // cols_per_row
    auth_table = doc.add_table(rows=num_rows, cols=cols_per_row)
    auth_table.alignment = 1
    _remove_table_borders(auth_table)

    for idx, author_name in enumerate(authors):
        cell = auth_table.cell(idx // cols_per_row, idx % cols_per_row)
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
            if len(lines) >= 2 and (lines[0] in subhead_candidates or (len(lines[0]) < 42 and not lines[0].endswith((".", ":", ",", ";")))):
                letter = sub_letters[sub_idx % len(sub_letters)]
                sub_idx += 1
                p_sub = doc.add_paragraph()
                p_sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p_sub.paragraph_format.space_before = Pt(5)
                p_sub.paragraph_format.space_after = Pt(2)
                p_sub.paragraph_format.keep_with_next = True
                rs = p_sub.add_run(f"{letter}. {lines[0]}")
                rs.font.name = "Times New Roman"
                rs.font.size = Pt(10)
                rs.italic = True
                first_para = True
                for bl in lines[1:]:
                    _add_ieee_body_paragraph(doc, bl, indent=not first_para and not bl.startswith(("-", "*", "•")))
                    first_para = False
            else:
                for bl in lines:
                    _add_ieee_body_paragraph(doc, bl, indent=not first_para and not bl.startswith(("-", "*", "•")))
                    first_para = False

        if sec_name == "Results and Discussion":
            _add_results_table_docx(doc, project, font_size_pt=8)

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


def _export_structured_conference_docx(project: ResearchProject, style_name: str) -> bytes:
    """Draft Word (.docx) formatted for Springer LNCS-style (1-col) or ACM-style layouts."""
    doc = docx.Document()
    sec0 = doc.sections[0]
    sec0.page_width = Inches(8.5)
    sec0.page_height = Inches(11.0)
    sec0.top_margin = Inches(1.0)
    sec0.bottom_margin = Inches(1.0)
    sec0.left_margin = Inches(1.0 if style_name == "Springer" else 0.75)
    sec0.right_margin = Inches(1.0 if style_name == "Springer" else 0.75)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)

    title = (project.proposal.title or project.selected_topic or project.title).strip()
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER if style_name == "Springer" else WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_after = Pt(8)
    rt = p_title.add_run(title)
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(16 if style_name == "Springer" else 18)
    rt.bold = True

    authors = ", ".join(project.publication_prep.authors or ["First Author"])
    affil = project.publication_prep.affiliation or "Department"
    p_meta = doc.add_paragraph()
    p_meta.alignment = p_title.alignment
    p_meta.paragraph_format.space_after = Pt(10)
    rm = p_meta.add_run(f"{authors}\n{affil}")
    rm.font.name = "Times New Roman"
    rm.font.size = Pt(10)
    rm.italic = True

    if style_name == "ACM":
        sec_body = doc.add_section(WD_SECTION.CONTINUOUS)
        sec_body.top_margin = Inches(0.85)
        sec_body.bottom_margin = Inches(0.85)
        sec_body.left_margin = Inches(0.75)
        sec_body.right_margin = Inches(0.75)
        _set_two_column_section(sec_body, num_cols=2, space_twips=360)

    sections = _collect_sections_preserving_approved(project)
    sec_dict = dict(sections)
    abstract_body = sec_dict.get("Abstract") or "Abstract pending."
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    ra_h = p_abs.add_run("Abstract. " if style_name == "Springer" else "ABSTRACT\n")
    ra_h.bold = True
    ra_h.font.size = Pt(9.5)
    ra_b = p_abs.add_run(abstract_body)
    ra_b.font.size = Pt(9.5)

    kw_str = ", ".join(project.publication_prep.keywords or [project.domain])
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(8)
    rk_h = p_kw.add_run("Keywords: " if style_name == "Springer" else "CCS Concepts & Keywords: ")
    rk_h.bold = True
    rk_h.font.size = Pt(9)
    rk_b = p_kw.add_run(kw_str)
    rk_b.font.size = Pt(9)

    sec_num = 0
    for sec_name, content in sections:
        if sec_name == "Abstract":
            continue
        if sec_name == "References":
            p_rh = doc.add_paragraph()
            p_rh.paragraph_format.space_before = Pt(10)
            rr = p_rh.add_run("References" if style_name == "Springer" else "REFERENCES")
            rr.bold = True
            rr.font.size = Pt(11)
            for line in [ln.strip() for ln in content.splitlines() if ln.strip()]:
                p_ref = doc.add_paragraph()
                p_ref.paragraph_format.left_indent = Inches(0.22)
                p_ref.paragraph_format.first_line_indent = Inches(-0.22)
                p_ref.paragraph_format.space_after = Pt(2)
                rt_ref = p_ref.add_run(line)
                rt_ref.font.size = Pt(9)
            continue

        sec_num += 1
        p_h = doc.add_paragraph()
        p_h.paragraph_format.space_before = Pt(10)
        p_h.paragraph_format.space_after = Pt(4)
        p_h.paragraph_format.keep_with_next = True
        rh = p_h.add_run(f"{sec_num}  {sec_name}" if style_name == "Springer" else f"{sec_num}  {sec_name.upper()}")
        rh.bold = True
        rh.font.size = Pt(11)

        for block in content.split("\n\n"):
            cleaned = re.sub(r"^#+\s*", "", block.strip())
            if cleaned:
                p_b = doc.add_paragraph(cleaned)
                p_b.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p_b.paragraph_format.space_after = Pt(4)

        if sec_name == "Results and Discussion":
            _add_results_table_docx(doc, project, font_size_pt=9)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def export_docx_bytes(project: ResearchProject) -> bytes:
    """Export the manuscript to a formatted Microsoft Word (.docx) binary matching the selected target style."""
    fmt = project.publication_prep.target_format
    if fmt == "IEEE":
        return _export_ieee_conference_docx(project)
    return _export_structured_conference_docx(project, fmt)


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
