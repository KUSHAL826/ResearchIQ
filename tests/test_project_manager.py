"""Tests for portable .riq project export, safe archive validation, import, and auto-save."""

from __future__ import annotations

import io
import zipfile

import pytest

from core.project_manager import (
    cleanup_project_temp_dir,
    create_empty_project,
    export_project_to_riq_bytes,
    is_safe_archive_member,
    session_autosave_backup,
    validate_and_import_riq_bytes,
)
from document_engine.chunking import process_pdf_bytes
from tests.test_document_engine import _make_sample_pdf_bytes


def test_safe_archive_member_validation() -> None:
    assert is_safe_archive_member("manifest.json") is True
    assert is_safe_archive_member("assets/pdfs/paper_1.pdf") is True
    assert is_safe_archive_member("../evil.txt") is False
    assert is_safe_archive_member("/etc/passwd") is False
    assert is_safe_archive_member("C:\\Windows\\system.ini") is False
    assert is_safe_archive_member("assets\\..\\secret.txt") is False


def test_reject_unsafe_riq_zip_traversal() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../outside.txt", b"malicious")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        validate_and_import_riq_bytes(buf.getvalue())


def test_export_and_import_riq_roundtrip() -> None:
    proj = create_empty_project(title="Portable Quantum AI Study", domain="Artificial Intelligence")
    pdf_bytes = _make_sample_pdf_bytes()
    parsed = process_pdf_bytes(pdf_bytes, "paper.pdf")
    rel = f"assets/pdfs/{parsed.paper_id}_paper.pdf"
    parsed.asset_relpath = rel
    proj.parsed_papers.append(parsed)
    proj.manuscript_sections["Abstract"].content = "Verified abstract draft."
    proj.manuscript_sections["Abstract"].approved = True
    proj.active_stage = "methodology"

    riq_bytes = export_project_to_riq_bytes(proj, asset_bytes_map={rel: pdf_bytes})
    restored, restored_assets, warnings = validate_and_import_riq_bytes(riq_bytes)

    try:
        assert not warnings
        assert restored.title == "Portable Quantum AI Study"
        assert restored.active_stage == "methodology"
        assert len(restored.parsed_papers) == 1
        assert restored.manuscript_sections["Abstract"].approved is True
        assert rel in restored_assets
        assert restored_assets[rel] == pdf_bytes
    finally:
        cleanup_project_temp_dir(proj.project_id)
        cleanup_project_temp_dir(restored.project_id)


def test_session_autosave_avoids_redundant_writes() -> None:
    proj = create_empty_project(title="Autosave Test")
    try:
        wrote1, digest1, _ = session_autosave_backup(proj, last_digest=None)
        wrote2, digest2, _ = session_autosave_backup(proj, last_digest=digest1)
        assert wrote1 is True
        assert wrote2 is False
        assert digest1 == digest2
    finally:
        cleanup_project_temp_dir(proj.project_id)
