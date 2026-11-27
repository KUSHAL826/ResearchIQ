"""Tests for Paper Studio Stage 9 real CSV results and Stage 12 DOCX/PDF/LaTeX/BibTeX exports."""

from __future__ import annotations

from core.project_manager import cleanup_project_temp_dir, create_empty_project
from core.schemas import ScholarlyWork
from paper_studio.publisher_exporter import (
    export_bibtex,
    export_docx_bytes,
    export_latex,
    export_pdf_bytes,
)
from paper_studio.results_analyzer import compute_classification_evaluation, register_uploaded_dataset


def test_stage9_real_csv_metrics_and_confusion_matrix() -> None:
    proj = create_empty_project(title="Classifier Benchmark")
    csv_bytes = (
        b"model,y_true,y_pred,accuracy\n"
        b"Proposed,1,1,0.92\n"
        b"Proposed,0,0,0.92\n"
        b"Proposed,1,1,0.92\n"
        b"Proposed,0,1,0.92\n"
    )
    try:
        ds_info, df = register_uploaded_dataset(proj, csv_bytes, "metrics.csv")
        assert ds_info.row_count == 4
        assert "accuracy" in ds_info.numeric_summary
        assert ds_info.numeric_summary["accuracy"]["mean"] == 0.92

        eval_res = compute_classification_evaluation(df, "y_true", "y_pred")
        assert eval_res["metrics"]["accuracy"] == 0.75
        assert len(eval_res["confusion_matrix"]) == 2
    finally:
        cleanup_project_temp_dir(proj.project_id)


def test_stage12_exports_preserve_approved_sections() -> None:
    import io
    import docx

    proj = create_empty_project(title="Export Integrity Study")
    proj.manuscript_sections["Abstract"].content = "Approved experimental abstract content."
    proj.manuscript_sections["Abstract"].approved = True
    proj.manuscript_sections["Introduction"].content = "Motivation\nAddressing dynamic environments.\n\nProblem Statement\nIntroduction grounded in prior work [1]."
    proj.manuscript_sections["Results and Discussion"].content = "Empirical accuracy improved to 0.92."
    proj.experiment_results.computed_classification_metrics = {
        "samples": 100,
        "accuracy": 0.92,
        "precision_macro": 0.91,
        "recall_macro": 0.9,
        "f1_macro": 0.905,
    }
    proj.discovered_works.append(
        ScholarlyWork(
            work_id="oa_99",
            source="openalex",
            title="Foundational Benchmark Paper",
            authors=["Jane Doe"],
            year=2023,
            venue="NeurIPS",
            doi="10.5555/12345",
            screening_status="included",
        )
    )

    bib = export_bibtex(proj)
    tex = export_latex(proj)
    docx_bytes = export_docx_bytes(proj)
    pdf_bytes = export_pdf_bytes(proj)

    assert "Foundational Benchmark Paper" in bib
    assert "Approved experimental abstract content." in tex
    assert "\\documentclass[conference]{IEEEtran}" in tex
    assert len(docx_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # Verify IEEE conference Word template structure: 2-column body, Abstract—, Index Terms—, Roman heads, TABLE I, REFERENCES
    parsed_doc = docx.Document(io.BytesIO(docx_bytes))
    assert len(parsed_doc.sections) == 2
    cols_el = parsed_doc.sections[1]._sectPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}cols")
    assert cols_el is not None
    assert cols_el.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num") == "2"

    all_text = "\n".join(p.text for p in parsed_doc.paragraphs)
    assert "Abstract—Approved experimental abstract content." in all_text
    assert "Index Terms—" in all_text
    assert "I. INTRODUCTION" in all_text
    assert "A. Motivation" in all_text
    assert "TABLE I" in all_text
    assert "REFERENCES" in all_text
    assert len(parsed_doc.tables) >= 2  # Author grid + TABLE I
