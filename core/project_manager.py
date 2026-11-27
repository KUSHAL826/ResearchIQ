"""Portable .riq project management, multi-project isolation, session auto-save, and safe archive import."""

from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Tuple

from core.config import (
    MANUSCRIPT_SECTIONS,
    MAX_RIQ_ARCHIVE_BYTES,
    RIQ_FORMAT_VERSION,
    get_app_temp_root,
)
from core.schemas import (
    ManuscriptSectionState,
    ResearchProject,
    utc_now_iso,
)


def sanitize_filename(name: str) -> str:
    """Strip unsafe characters and directory components from a filename."""
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "file.bin"


def is_safe_archive_member(member_name: str) -> bool:
    """Reject absolute paths, Windows drive paths, backslashes escaping, and path traversal."""
    if not member_name or member_name.startswith("/") or member_name.startswith("\\"):
        return False
    if "\\" in member_name:
        return False
    if len(member_name) >= 2 and member_name[1] == ":":
        return False
    norm = posixpath.normpath(member_name)
    if norm.startswith("..") or norm == "." or "/../" in f"/{norm}/":
        return False
    parts = PurePosixPath(member_name).parts
    if any(part in ("..", "") for part in parts):
        return False
    return True


def create_empty_project(title: str = "Untitled Research Project", domain: str = "Artificial Intelligence") -> ResearchProject:
    """Create a fresh ResearchProject with initialized 12 manuscript sections."""
    proj = ResearchProject(title=title, domain=domain)
    proj.manuscript_sections = {
        sec: ManuscriptSectionState(name=sec, content="", approved=False)
        for sec in MANUSCRIPT_SECTIONS
    }
    return proj


def ensure_project_sections(project: ResearchProject) -> ResearchProject:
    """Ensure all 12 standard manuscript sections exist in project state."""
    for sec in MANUSCRIPT_SECTIONS:
        if sec not in project.manuscript_sections:
            project.manuscript_sections[sec] = ManuscriptSectionState(name=sec)
    return project


def get_project_temp_dir(project_id: str) -> Path:
    """Return an isolated temporary directory for the given project ID."""
    safe_id = sanitize_filename(project_id)
    path = get_app_temp_root() / safe_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "pdfs").mkdir(exist_ok=True)
    (path / "datasets").mkdir(exist_ok=True)
    (path / "diagrams").mkdir(exist_ok=True)
    return path


def cleanup_project_temp_dir(project_id: str) -> None:
    """Delete all temporary files associated with a project ID."""
    safe_id = sanitize_filename(project_id)
    path = get_app_temp_root() / safe_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def compute_project_digest(project: ResearchProject) -> str:
    """Compute a deterministic hash of project state to avoid redundant auto-save writes."""
    payload = project.model_dump(exclude={"updated_at", "unsaved_changes"})
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def session_autosave_backup(
    project: ResearchProject,
    last_digest: str | None = None,
) -> Tuple[bool, str, str]:
    """
    Write a temporary session backup JSON only if state actually changed.
    Returns (wrote_backup, new_digest, backup_timestamp).
    """
    current_digest = compute_project_digest(project)
    if last_digest and current_digest == last_digest:
        return False, current_digest, ""

    temp_dir = get_project_temp_dir(project.project_id)
    backup_path = temp_dir / "session_autosave.json"
    ts = utc_now_iso()
    data = {
        "autosave_timestamp": ts,
        "format_version": RIQ_FORMAT_VERSION,
        "note": "Temporary session backup only. Export a .riq file for permanent local storage.",
        "project": project.model_dump(mode="json"),
    }
    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True, current_digest, ts


def mark_project_exported(project: ResearchProject) -> None:
    """Update export metadata only when the user actually triggers a .riq download."""
    export_time = utc_now_iso()
    project.last_exported_at = export_time
    project.unsaved_changes = False
    project.updated_at = export_time


def export_project_to_riq_bytes(
    project: ResearchProject,
    asset_bytes_map: dict[str, bytes] | None = None,
    mark_exported: bool = True,
) -> bytes:
    """
    Package the entire ResearchProject and its uploaded PDFs, datasets, and diagrams
    into a portable .riq ZIP archive.
    """
    project = ensure_project_sections(project)
    export_time = utc_now_iso()
    if mark_exported:
        project.last_exported_at = export_time
        project.unsaved_changes = False
        project.updated_at = export_time

    temp_dir = get_project_temp_dir(project.project_id)
    assets_to_write: dict[str, bytes] = {}

    # Collect uploaded PDFs
    for paper in project.parsed_papers:
        rel = paper.asset_relpath or f"assets/pdfs/{paper.paper_id}_{sanitize_filename(paper.filename)}"
        paper.asset_relpath = rel
        if asset_bytes_map and rel in asset_bytes_map:
            assets_to_write[rel] = asset_bytes_map[rel]
        else:
            disk_candidate = temp_dir / "pdfs" / Path(rel).name
            if disk_candidate.exists():
                assets_to_write[rel] = disk_candidate.read_bytes()

    # Collect uploaded datasets
    for ds in project.experiment_results.datasets:
        rel = ds.asset_relpath or f"assets/datasets/{ds.dataset_id}_{sanitize_filename(ds.filename)}"
        ds.asset_relpath = rel
        if asset_bytes_map and rel in asset_bytes_map:
            assets_to_write[rel] = asset_bytes_map[rel]
        else:
            disk_candidate = temp_dir / "datasets" / Path(rel).name
            if disk_candidate.exists():
                assets_to_write[rel] = disk_candidate.read_bytes()

    # Collect saved diagrams (Mermaid/SVG)
    for diag_name, diag_src in project.saved_diagrams.items():
        if diag_src:
            rel = f"assets/diagrams/{sanitize_filename(diag_name)}.mmd"
            assets_to_write[rel] = diag_src.encode("utf-8")

    manifest: dict[str, Any] = {
        "riq_format_version": RIQ_FORMAT_VERSION,
        "application": "RESEARCHIQ",
        "exported_at": export_time,
        "project_id": project.project_id,
        "project_title": project.title,
        "included_assets": sorted(assets_to_write.keys()),
        "project": project.model_dump(mode="json"),
    }

    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        for rel_path, content in assets_to_write.items():
            if is_safe_archive_member(rel_path):
                zf.writestr(rel_path, content)
    return buf.getvalue()


def validate_and_import_riq_bytes(
    raw_bytes: bytes,
) -> Tuple[ResearchProject, dict[str, bytes], list[str]]:
    """
    Validate and import a .riq ZIP archive safely.
    Returns (restored_project, restored_assets_map, warnings).
    Raises ValueError on invalid or unsafe archives.
    """
    if not raw_bytes:
        raise ValueError("Uploaded .riq file is empty.")
    if len(raw_bytes) > MAX_RIQ_ARCHIVE_BYTES:
        raise ValueError(f".riq archive exceeds maximum allowed size ({MAX_RIQ_ARCHIVE_BYTES // (1024 * 1024)} MB).")

    buf = io.BytesIO(raw_bytes)
    if not zipfile.is_zipfile(buf):
        raise ValueError("Invalid .riq file: archive is not a valid ZIP package.")

    buf.seek(0)
    warnings: list[str] = []
    restored_assets: dict[str, bytes] = {}

    with zipfile.ZipFile(buf, mode="r") as zf:
        infos = zf.infolist()
        if not infos:
            raise ValueError("Invalid .riq archive: no entries found.")

        total_uncompressed = 0
        for info in infos:
            if not is_safe_archive_member(info.filename):
                raise ValueError(f"Unsafe archive path detected and rejected: '{info.filename}'")
            # Reject symlinks (external_attr Unix symlink bit 0o120000)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symbolic links are not allowed in .riq archives: '{info.filename}'")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_RIQ_ARCHIVE_BYTES * 2:
                raise ValueError("Archive uncompressed size exceeds safety limit.")

        names = {info.filename for info in infos}
        if "manifest.json" not in names:
            raise ValueError("Invalid .riq archive: missing required 'manifest.json'.")

        try:
            manifest_raw = zf.read("manifest.json").decode("utf-8")
            manifest = json.loads(manifest_raw)
        except Exception as exc:
            raise ValueError(f"Corrupted manifest.json in .riq archive: {exc}") from exc

        if "project" not in manifest or not isinstance(manifest["project"], dict):
            raise ValueError("Invalid manifest.json: missing 'project' payload.")

        try:
            project = ResearchProject.model_validate(manifest["project"])
        except Exception as exc:
            raise ValueError(f"Project schema validation failed: {exc}") from exc

        project = ensure_project_sections(project)
        project.unsaved_changes = False

        # Restore assets to isolated temp directory
        temp_dir = get_project_temp_dir(project.project_id)
        for info in infos:
            if info.is_dir() or info.filename == "manifest.json":
                continue
            data = zf.read(info.filename)
            restored_assets[info.filename] = data
            if info.filename.startswith("assets/pdfs/"):
                out_file = temp_dir / "pdfs" / Path(info.filename).name
                out_file.write_bytes(data)
            elif info.filename.startswith("assets/datasets/"):
                out_file = temp_dir / "datasets" / Path(info.filename).name
                out_file.write_bytes(data)
            elif info.filename.startswith("assets/diagrams/"):
                out_file = temp_dir / "diagrams" / Path(info.filename).name
                out_file.write_bytes(data)

        for rel in manifest.get("included_assets", []):
            if rel not in restored_assets:
                warnings.append(f"Listed asset '{rel}' was not present in the archive.")

    return project, restored_assets, warnings
