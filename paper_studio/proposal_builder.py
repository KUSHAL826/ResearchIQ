"""Studio Stage 5: Research Proposal and Milestone Roadmap Builder."""

from __future__ import annotations

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ProposalData, ResearchProject


def generate_research_proposal(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> ProposalData:
    """Generate a structured research proposal from approved upstream stages."""
    approved_ctx = build_approved_context_summary(project)
    ps = project.problem_statement
    title = project.selected_topic or project.title

    if llm_client and llm_client.is_configured:
        prompt = (
            f"Create a structured Research Proposal (ProposalData) grounded in the current project context.\n"
            f"APPROVED CONTEXT:\n{approved_ctx}\n\n"
            f"Problem Definition: {ps.problem_definition}\n"
            f"Motivation: {ps.motivation}\n"
            f"Objectives: {ps.objectives}\n"
            f"Research Questions: {ps.research_questions}\n"
            f"Include 5 realistic research milestones as list of dicts with keys: 'phase', 'deliverable', 'duration'. "
            f"Keep approved=False."
        )
        try:
            res = llm_client.generate_structured(prompt, ProposalData, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            res.approved = False
            return res
        except Exception:
            pass

    return ProposalData(
        title=title,
        background=(
            f"Research in {project.domain} increasingly relies on automated pipelines, yet "
            f"{len(project.research_gaps)} gap(s) identified across {len(project.parsed_papers)} analysed paper(s) "
            f"and {len([w for w in project.discovered_works if not w.is_duplicate])} discovered works highlight open challenges."
        ),
        problem_statement=ps.problem_definition or f"Investigating robust methods for {title}.",
        motivation=ps.motivation or "Improving reliability, reproducibility, and empirical rigor.",
        objectives=list(ps.objectives) or ["O1: Establish reproducible baselines", "O2: Validate proposed architecture"],
        research_questions=list(ps.research_questions) or ["RQ1: How does the proposed method improve target metrics?"],
        methodology_summary=(
            "A staged pipeline combining curated preprocessing, modular model design, "
            "baseline benchmarking, and ablation-driven statistical validation."
        ),
        expected_contributions=[
            g.potential_improvement for g in project.research_gaps[:3] if g.potential_improvement
        ] or [
            "Reproducible experimental framework",
            "Comprehensive baseline and ablation benchmarks",
        ],
        scope="; ".join(ps.scope_in) or f"Focused empirical validation within {project.domain}.",
        milestones=[
            {"phase": "Phase 1: Literature & Gap Audit", "deliverable": "PRISMA Matrix & Gap Map", "duration": "Weeks 1-2"},
            {"phase": "Phase 2: Architecture & Data Prep", "deliverable": "Pipeline Spec & Clean Splits", "duration": "Weeks 3-4"},
            {"phase": "Phase 3: Core Implementation", "deliverable": "Validated Prototype & Unit Tests", "duration": "Weeks 5-7"},
            {"phase": "Phase 4: Experiments & Ablation", "deliverable": "CSV Metrics & Error Analysis", "duration": "Weeks 8-10"},
            {"phase": "Phase 5: Manuscript & Review", "deliverable": "Camera-Ready IEEE/ACM Draft", "duration": "Weeks 11-12"},
        ],
        approved=False,
    )
