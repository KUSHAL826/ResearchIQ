"""Application configuration and constants for RESEARCHIQ."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "RESEARCHIQ"
APP_TAGLINE = "Visual-First Autonomous Research Intelligence & Manuscript Studio"
RIQ_FORMAT_VERSION = "1.0.0"

DEFAULT_LLM_MODEL = os.getenv("RESEARCHIQ_LLM_MODEL", "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("RESEARCHIQ_EMBEDDING_MODEL", "text-embedding-004")

OPENALEX_BASE_URL = "https://api.openalex.org"
ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"

WORKFLOW_STAGES = [
    {"id": "idea", "label": "IDEA", "title": "Research Idea", "studio_stage": 1},
    {"id": "discovery", "label": "DISCOVERY", "title": "Discovery & Problem", "studio_stage": 2},
    {"id": "literature_review", "label": "LITERATURE REVIEW", "title": "Literature Review", "studio_stage": 3},
    {"id": "gap_identification", "label": "GAP IDENTIFICATION", "title": "Research Gap & Contribution", "studio_stage": 4},
    {"id": "proposal", "label": "PROPOSAL", "title": "Research Proposal", "studio_stage": 5},
    {"id": "methodology", "label": "METHODOLOGY", "title": "Methodology & Architecture", "studio_stage": 6},
    {"id": "implementation", "label": "IMPLEMENTATION", "title": "Implementation", "studio_stage": 7},
    {"id": "experiments", "label": "EXPERIMENTS", "title": "Experiment Design", "studio_stage": 8},
    {"id": "results", "label": "RESULTS", "title": "Results & Discussion", "studio_stage": 9},
    {"id": "manuscript", "label": "MANUSCRIPT", "title": "Manuscript Generation", "studio_stage": 10},
    {"id": "review", "label": "REVIEW", "title": "Manuscript Validation", "studio_stage": 11},
    {"id": "final_paper", "label": "FINAL PAPER", "title": "Publication Preparation", "studio_stage": 12},
]

STANDARD_PAPER_SECTIONS = [
    "Abstract",
    "Introduction",
    "Literature Review",
    "Methodology",
    "Experiments",
    "Results",
    "Conclusion",
]

MANUSCRIPT_SECTIONS = [
    "Abstract",
    "Introduction",
    "Literature Review",
    "Research Gap",
    "Objectives",
    "Proposed Methodology",
    "Implementation",
    "Experimental Setup",
    "Results and Discussion",
    "Conclusion",
    "Future Scope",
    "References",
]

PRIVACY_DISCLOSURE = (
    "Session-Only Privacy Notice: RESEARCHIQ has no login, user accounts, or permanent "
    "server database. All research context lives in your active session and temporary files. "
    "Documents, prompts, and queries sent for AI analysis or scholarly lookup are processed by "
    "Google Gemini and external academic APIs (OpenAlex, arXiv, Semantic Scholar) under their "
    "respective terms. Export a .riq project file to save your work locally."
)

MAX_RIQ_ARCHIVE_BYTES = 200 * 1024 * 1024  # 200 MB safety cap
MAX_UPLOAD_PDF_BYTES = 50 * 1024 * 1024    # 50 MB per PDF


def get_gemini_api_key(session_override: str | None = None) -> str:
    """Resolve Gemini API key from session override or environment."""
    if session_override and session_override.strip():
        return session_override.strip()
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


def get_app_temp_root() -> Path:
    """Return a clean session-scoped temporary root directory."""
    import tempfile

    root = Path(tempfile.gettempdir()) / "researchiq_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root
