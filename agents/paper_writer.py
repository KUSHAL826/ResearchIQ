"""Section-wise academic manuscript writer with evidence injection and formatting helpers."""

from __future__ import annotations

import re
from typing import Any

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SECTION_WRITING_GUIDANCE, SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ResearchProject


def build_reference_list(project: ResearchProject, style: str = "IEEE") -> list[str]:
    """Compile real formatted references strictly from session PDFs and included/discovered works."""
    refs: list[str] = []
    seen_titles: set[str] = set()

    for p in project.parsed_papers:
        key = re.sub(r"[^a-z0-9]", "", p.title.lower())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        authors = ", ".join(p.authors[:4]) if p.authors else "Authors Not Parsed"
        year = p.year or "n.d."
        doi_str = f" doi: {p.doi}." if p.doi else ""
        refs.append(f"[{len(refs) + 1}] {authors}, \"{p.title},\" {year}.{doi_str}")

    included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
    pool = included if included else [w for w in project.discovered_works if not w.is_duplicate][:15]
    for w in pool:
        key = re.sub(r"[^a-z0-9]", "", w.title.lower())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        authors = ", ".join(w.authors[:4]) if w.authors else "Unknown Author"
        venue = w.venue or w.source
        year = w.year or "n.d."
        doi_str = f" doi: {w.doi}." if w.doi else (f" {w.url}" if w.url else "")
        if style == "ACM":
            refs.append(f"[{len(refs) + 1}] {authors}. {year}. {w.title}. In {venue}.{doi_str}")
        elif style == "Springer":
            refs.append(f"{len(refs) + 1}. {authors}: {w.title}. {venue} ({year}).{doi_str}")
        else:
            refs.append(f"[{len(refs) + 1}] {authors}, \"{w.title},\" {venue}, {year}.{doi_str}")

    return refs


def draft_manuscript_section(
    project: ResearchProject,
    section_name: str,
    retrieved_evidence: list[dict[str, Any]],
    llm_client: GeminiClient | None = None,
    instruction_override: str = "",
) -> str:
    """Draft a single manuscript section grounded in approved project stages and session evidence."""
    if section_name == "References":
        refs = build_reference_list(project, style=project.publication_prep.target_format)
        return "\n\n".join(refs) if refs else "No verified papers or scholarly works have been added to this workspace yet."

    approved_ctx = build_approved_context_summary(project)
    guidance = SECTION_WRITING_GUIDANCE.get(section_name, "Write a rigorous academic section.")

    evidence_block = "\n".join(
        f"- [{e['paper_title']}, {e['pages']}]: {e['text'][:350]}"
        for e in retrieved_evidence[:5]
    ) or "No uploaded PDF excerpts matched."

    # Actual experimental metrics block for Results and Discussion
    metrics_block = ""
    if project.experiment_results.datasets:
        ds = project.experiment_results.datasets[0]
        metrics_block = (
            f"Uploaded Dataset: {ds.filename} ({ds.row_count} rows, columns={ds.columns})\n"
            f"Numeric Summary: {ds.numeric_summary}\n"
            f"Computed Metrics: {project.experiment_results.computed_classification_metrics}"
        )

    if llm_client and llm_client.is_configured:
        prompt = (
            f"Draft the '{section_name}' section for an academic research paper.\n"
            f"SECTION GOAL: {guidance}\n"
            f"EXTRA USER INSTRUCTION: {instruction_override or 'None'}\n\n"
            f"APPROVED PROJECT CONTEXT:\n{approved_ctx}\n\n"
            f"PROBLEM DEFINITION:\n{project.problem_statement.problem_definition}\n"
            f"METHODOLOGY:\n{project.methodology.proposed_architecture}\n"
            f"IMPLEMENTATION NOTES:\n{project.implementation.documentation_notes}\n"
            f"EXPERIMENT DESIGN:\nMetrics={project.experiment_design.evaluation_metrics}\n"
            f"VERIFIED EXPERIMENTAL RESULTS (NEVER INVENT NUMBERS):\n{metrics_block or 'None uploaded yet'}\n\n"
            f"RETRIEVED LITERATURE EVIDENCE:\n{evidence_block}\n\n"
            f"Return clean academic prose in markdown without inventing any statistics or citations."
        )
        try:
            return llm_client.generate_text(prompt, system_instruction=SYSTEM_RESEARCH_ARCHITECT, temperature=0.25)
        except Exception:
            pass

    return _offline_section_draft(project, section_name, metrics_block, retrieved_evidence)


def improve_or_rewrite_section(
    section_name: str,
    current_text: str,
    action: str,  # "improve" | "rewrite" | "add_evidence"
    retrieved_evidence: list[dict[str, Any]],
    llm_client: GeminiClient | None = None,
    custom_note: str = "",
) -> str:
    """Refine, rewrite, or inject retrieved citations into an existing section draft."""
    evidence_block = "\n".join(
        f"- [{e['paper_title']}, {e['pages']}]: {e['text'][:320]}"
        for e in retrieved_evidence[:4]
    )
    if llm_client and llm_client.is_configured and current_text.strip():
        verb = {
            "improve": "Improve academic clarity, flow, and precision while preserving all factual claims and numbers.",
            "rewrite": "Rewrite this content with stronger scholarly structure and concise technical phrasing.",
            "add_evidence": "Integrate relevant supporting evidence from the retrieved passages below with explicit source attributions.",
        }.get(action, "Refine this academic text.")
        prompt = (
            f"Section: {section_name}\nAction: {verb}\nAdditional note: {custom_note or 'None'}\n\n"
            f"CURRENT TEXT:\n{current_text}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_block or 'None'}"
        )
        try:
            return llm_client.generate_text(prompt, system_instruction=SYSTEM_RESEARCH_ARCHITECT, temperature=0.2)
        except Exception:
            pass

    if action == "add_evidence" and retrieved_evidence:
        addendum = "\n\n**Supporting Session Evidence:**\n" + "\n".join(
            f"- According to *{e['paper_title']}* ({e['pages']}): \"{e['text'][:180]}...\""
            for e in retrieved_evidence[:3]
        )
        return current_text + addendum
    return current_text


def _offline_section_draft(
    project: ResearchProject,
    section_name: str,
    metrics_block: str,
    retrieved_evidence: list[dict[str, Any]],
) -> str:
    title = project.selected_topic or project.title
    if section_name == "Abstract":
        return (
            f"This study investigates {title} within the {project.domain} domain. "
            f"{project.problem_statement.problem_definition or 'We address key methodological limitations identified in current literature.'} "
            f"Building on {len(project.research_gaps)} identified research gap(s), we design a structured methodology "
            f"evaluated using {', '.join(project.experiment_design.evaluation_metrics) or 'standard quantitative benchmarks'}."
        )
    if section_name == "Introduction":
        objs = "\n".join(f"- {o}" for o in project.problem_statement.objectives) or "- Objective 1: Formulate and validate the proposed framework."
        return f"### Context & Motivation\n{project.problem_statement.motivation or project.problem_statement.problem_definition or title}\n\n### Objectives\n{objs}"
    if section_name == "Literature Review":
        works = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"] or project.discovered_works[:5]
        lines = [f"- **{w.title}** ({w.year or 'n.d.'}): {w.extracted_contribution or w.abstract[:180]}" for w in works[:8]]
        return "### Thematic Synthesis\n" + ("\n".join(lines) if lines else "Add or screen papers in Literature Review to synthesize findings.")
    if section_name == "Research Gap":
        lines = [f"- **{g.title}** [{g.classification}, {g.supporting_publication}, {g.page_reference}]: {g.description}" for g in project.research_gaps]
        return "\n".join(lines) if lines else "Run the Research Gap Detective to populate evidence-backed gaps."
    if section_name == "Objectives":
        return "\n".join(f"{i + 1}. {o}" for i, o in enumerate(project.problem_statement.objectives)) or "Define objectives in Stage 2."
    if section_name == "Proposed Methodology":
        return project.methodology.proposed_architecture or project.methodology.research_design or "Complete Stage 6 to generate architecture details."
    if section_name == "Implementation":
        return project.implementation.documentation_notes or "Complete Stage 7 to document implementation modules."
    if section_name == "Experimental Setup":
        metrics = ", ".join(project.experiment_design.evaluation_metrics) or "Pending metric selection"
        return f"**Evaluation Metrics:** {metrics}\n\n**Baselines:** {len(project.experiment_design.baselines)} configured."
    if section_name == "Results and Discussion":
        return metrics_block or project.experiment_results.discussion_notes or "Upload CSV/Excel experimental results in Stage 9 to generate verified result analysis."
    if section_name == "Conclusion":
        return f"This work presented {title}. Future validation should address the documented limitations and extend cross-dataset benchmarks."
    if section_name == "Future Scope":
        return "Future extensions include multi-domain benchmarking, robustness calibration, and deployment-scale efficiency evaluation."
    return ""
