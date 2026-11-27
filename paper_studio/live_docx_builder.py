"""Live Word (.docx) Manuscript Builder: incrementally syncs Workspace B components into the active Word document."""

from __future__ import annotations

from typing import Any

from agents.paper_writer import build_reference_list
from core.config import MANUSCRIPT_SECTIONS
from core.project_manager import get_project_temp_dir
from core.schemas import ManuscriptSectionState, ResearchProject, utc_now_iso
from paper_studio.publisher_exporter import export_docx_bytes


def _set_section_text(
    project: ResearchProject,
    section_name: str,
    new_content: str,
    force: bool = False,
) -> bool:
    """Update a manuscript section unless already approved (locked) by the user, unless force=True."""
    sec = project.manuscript_sections.setdefault(section_name, ManuscriptSectionState(name=section_name))
    if sec.approved and not force:
        return False
    cleaned = new_content.strip()
    if not cleaned or sec.content.strip() == cleaned:
        return False
    sec.content = cleaned
    sec.word_count = len(cleaned.split())
    sec.last_updated = utc_now_iso()
    return True


def rebuild_live_docx(
    project: ResearchProject,
    assets_cache: dict[str, bytes] | None = None,
) -> bytes:
    """Generate fresh .docx bytes from current manuscript sections and store in session temp/cache."""
    docx_bytes = export_docx_bytes(project)
    temp_dir = get_project_temp_dir(project.project_id)
    (temp_dir / "live_manuscript.docx").write_bytes(docx_bytes)
    if assets_cache is not None:
        assets_cache["assets/live_manuscript.docx"] = docx_bytes
    return docx_bytes


def sync_stage_to_word_doc(
    project: ResearchProject,
    stage_key: str,
    assets_cache: dict[str, bytes] | None = None,
    force: bool = False,
) -> list[str]:
    """
    Sync structured content from a Workspace B stage into the corresponding Word manuscript sections
    and rebuild the live .docx file. Returns the list of updated section names.
    """
    updated: list[str] = []
    title = project.proposal.title or project.selected_topic or project.title

    if stage_key in ("title", "all"):
        if not project.manuscript_sections.get("Abstract", ManuscriptSectionState(name="Abstract")).content.strip():
            init_abs = (
                f"This paper investigates {title} in the domain of {project.domain}. "
                f"Subsequent sections formalize the problem statement, synthesis of prior literature, "
                f"identified research gaps, proposed methodology, experimental protocol, and verified empirical findings."
            )
            if _set_section_text(project, "Abstract", init_abs, force=force):
                updated.append("Abstract")

    if stage_key in ("problem", "all"):
        ps = project.problem_statement
        if ps.problem_definition or ps.motivation:
            intro_parts = []
            if ps.motivation:
                intro_parts.append(f"Motivation\n{ps.motivation}")
            if ps.problem_definition:
                intro_parts.append(f"Problem Statement\n{ps.problem_definition}")
            if ps.scope_in:
                intro_parts.append("Scope\n" + "\n".join(f"- {s}" for s in ps.scope_in))
            if _set_section_text(project, "Introduction", "\n\n".join(intro_parts), force=force):
                updated.append("Introduction")

        obj_parts = []
        if ps.objectives:
            obj_parts.append("Research Objectives\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(ps.objectives)))
        if ps.research_questions:
            obj_parts.append("Research Questions\n" + "\n".join(f"- {rq}" for rq in ps.research_questions))
        if ps.hypotheses:
            obj_parts.append("Hypotheses\n" + "\n".join(f"- {h}" for h in ps.hypotheses))
        if obj_parts and _set_section_text(project, "Objectives", "\n\n".join(obj_parts), force=force):
            updated.append("Objectives")

    if stage_key in ("literature", "all"):
        included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
        pool = included or [w for w in project.discovered_works if not w.is_duplicate][:8]
        lit_lines = []
        for idx, w in enumerate(pool, start=1):
            contrib = w.extracted_contribution or w.extracted_methodology or (w.abstract[:220] if w.abstract else "")
            lit_lines.append(f"[{idx}] {w.title} ({w.year or 'n.d.'}): {contrib}")
        for p in project.parsed_papers:
            lit_lines.append(f"- Uploaded Study: {p.title} ({p.year or 'n.d.'}) — {p.hierarchical_summary[:220]}")
        if lit_lines and _set_section_text(project, "Literature Review", "\n\n".join(lit_lines), force=force):
            updated.append("Literature Review")

        refs = build_reference_list(project, style=project.publication_prep.target_format)
        if refs and _set_section_text(project, "References", "\n\n".join(refs), force=force):
            updated.append("References")

    if stage_key in ("gaps", "all") and project.research_gaps:
        gap_lines = []
        for idx, g in enumerate(project.research_gaps, start=1):
            gap_lines.append(
                f"Gap {idx} ({g.classification.upper()} | {g.supporting_publication}, {g.page_reference}): "
                f"{g.title} — {g.description} Proposed improvement: {g.potential_improvement}"
            )
        if _set_section_text(project, "Research Gap", "\n\n".join(gap_lines), force=force):
            updated.append("Research Gap")

    if stage_key in ("proposal", "all"):
        prop = project.proposal
        if prop.background or prop.methodology_summary:
            abs_text = (
                f"This study addresses {prop.title or title}. "
                f"{prop.problem_statement} "
                f"Methodology overview: {prop.methodology_summary} "
                f"Expected contributions: {'; '.join(prop.expected_contributions[:3])}."
            )
            if _set_section_text(project, "Abstract", abs_text, force=force):
                updated.append("Abstract")

    if stage_key in ("methodology", "all"):
        md = project.methodology
        meth_parts = []
        if md.research_design:
            meth_parts.append(f"Research Design\n{md.research_design}")
        if md.proposed_architecture:
            meth_parts.append(f"Proposed Architecture\n{md.proposed_architecture}")
        if md.data_flow_description:
            meth_parts.append(f"Data Flow\n{md.data_flow_description}")
        if md.mathematical_models:
            meth_parts.append(f"Mathematical Formulation\n{md.mathematical_models}")
        if md.algorithms:
            alg_lines = "\n".join(
                f"- {a.get('name', 'Module')}: {a.get('role', '')} (Complexity: {a.get('complexity', 'N/A')})"
                for a in md.algorithms
            )
            meth_parts.append(f"Algorithms & Components\n{alg_lines}")
        if meth_parts and _set_section_text(project, "Proposed Methodology", "\n\n".join(meth_parts), force=force):
            updated.append("Proposed Methodology")

    if stage_key in ("implementation", "all"):
        imp = project.implementation
        imp_parts = []
        if imp.tech_stack:
            ts_lines = "\n".join(f"- {t.get('layer', '')}: {t.get('tool', '')} ({t.get('rationale', '')})" for t in imp.tech_stack)
            imp_parts.append(f"Technology Stack\n{ts_lines}")
        if imp.preprocessing_steps:
            imp_parts.append("Preprocessing Steps\n" + "\n".join(f"- {s}" for s in imp.preprocessing_steps))
        if imp.modules:
            mod_lines = "\n".join(f"- {m.get('module', '')}: {m.get('responsibility', '')}" for m in imp.modules)
            imp_parts.append(f"Module Organization\n{mod_lines}")
        if imp.documentation_notes:
            imp_parts.append(f"Reproducibility Notes\n{imp.documentation_notes}")
        if imp_parts and _set_section_text(project, "Implementation", "\n\n".join(imp_parts), force=force):
            updated.append("Implementation")

    if stage_key in ("experiments", "all"):
        ed = project.experiment_design
        exp_parts = []
        if ed.baselines:
            bl_lines = "\n".join(f"- {b.get('name', '')} ({b.get('type', '')}): {b.get('rationale', '')}" for b in ed.baselines)
            exp_parts.append(f"Baselines\n{bl_lines}")
        if ed.evaluation_metrics:
            exp_parts.append("Evaluation Metrics\n" + ", ".join(ed.evaluation_metrics))
        if ed.configurations:
            cfg_lines = "\n".join(f"- {c.get('parameter', '')}: {c.get('values', '')} ({c.get('notes', '')})" for c in ed.configurations)
            exp_parts.append(f"Configurations\n{cfg_lines}")
        if ed.ablation_plan:
            abl_lines = "\n".join(f"- {a.get('variant', '')}: {a.get('removed_or_changed', '')} — {a.get('hypothesis', '')}" for a in ed.ablation_plan)
            exp_parts.append(f"Ablation Plan\n{abl_lines}")
        if exp_parts and _set_section_text(project, "Experimental Setup", "\n\n".join(exp_parts), force=force):
            updated.append("Experimental Setup")

    if stage_key in ("results", "all"):
        er = project.experiment_results
        res_parts = []
        if er.datasets:
            for ds in er.datasets:
                stats_str = "; ".join(
                    f"{col} (mean={vals.get('mean')}, std={vals.get('std')}, min={vals.get('min')}, max={vals.get('max')})"
                    for col, vals in ds.numeric_summary.items()
                )
                res_parts.append(f"Dataset '{ds.filename}' ({ds.row_count} rows): {stats_str or 'Categorical columns loaded'}")
        if er.computed_classification_metrics:
            m = er.computed_classification_metrics
            res_parts.append(
                f"Computed Classification Metrics (N={m.get('samples')}): "
                f"Accuracy={m.get('accuracy')}, Macro Precision={m.get('precision_macro')}, "
                f"Macro Recall={m.get('recall_macro')}, Macro F1={m.get('f1_macro')}."
            )
        if er.discussion_notes.strip():
            res_parts.append(f"Discussion\n{er.discussion_notes.strip()}")
        if res_parts and _set_section_text(project, "Results and Discussion", "\n\n".join(res_parts), force=force):
            updated.append("Results and Discussion")

        if er.datasets and not project.manuscript_sections.get("Conclusion", ManuscriptSectionState(name="Conclusion")).content.strip():
            conc = (
                f"This study investigated {title}. Empirical evaluation on uploaded benchmark records "
                f"verified the proposed workflow, while highlighting areas for future multi-domain scaling."
            )
            if _set_section_text(project, "Conclusion", conc, force=force):
                updated.append("Conclusion")

    if updated or stage_key in ("title", "all"):
        project.unsaved_changes = True
        project.updated_at = utc_now_iso()
    rebuild_live_docx(project, assets_cache=assets_cache)
    return updated


def get_word_doc_summary(project: ResearchProject) -> dict[str, Any]:
    """Return populated sections and word totals for the Live Word Doc assistant."""
    rows = []
    total_words = 0
    populated = 0
    for sec_name in MANUSCRIPT_SECTIONS:
        sec = project.manuscript_sections.get(sec_name, ManuscriptSectionState(name=sec_name))
        words = len(sec.content.split()) if sec.content.strip() else 0
        total_words += words
        if words > 0:
            populated += 1
        rows.append(
            {
                "section": sec_name,
                "words": words,
                "approved": sec.approved,
                "populated": words > 0,
            }
        )
    return {
        "total_words": total_words,
        "populated_count": populated,
        "total_sections": len(MANUSCRIPT_SECTIONS),
        "rows": rows,
    }
