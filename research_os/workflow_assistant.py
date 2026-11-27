"""Part 9: Contextual Project-Level Research Workflow Assistant powered by LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ResearchProject
from core.vector_store import SessionVectorStore
from paper_studio.manuscript_manager import compute_manuscript_completion
from research.consistency_checker import run_consistency_check
from research.systematic_review import compute_prisma_counts


class _AssistantState(TypedDict, total=False):
    question: str
    diagnostics: dict[str, Any]
    rag_hits: list[dict[str, Any]]
    answer: str
    recommended_actions: list[str]


def compute_project_diagnostics(project: ResearchProject) -> dict[str, Any]:
    """Assemble live structured state across all research stages without altering approved content."""
    prisma = compute_prisma_counts(project.discovered_works, project.parsed_papers)
    comp = compute_manuscript_completion(project)
    consistency = run_consistency_check(project)

    strongest_gaps = sorted(
        project.research_gaps,
        key=lambda g: (1 if g.classification == "explicit" else 0, g.impact_score + g.feasibility_score),
        reverse=True,
    )
    unsupported_claims = [
        c for c in project.claim_verifications if c.support_status in ("Not supported", "Unable to verify")
    ]
    pending_exps = [r for r in project.experiment_records if r.status == "Planned"]

    # Shared methodologies across uploaded papers
    alg_papers: dict[str, list[str]] = {}
    for a in project.paper_analyses.values():
        for alg in a.algorithms:
            alg_papers.setdefault(alg.name, []).append(a.paper_title[:40])
    shared_methods = {k: v for k, v in alg_papers.items() if len(v) >= 2}

    return {
        "title": project.title,
        "domain": project.domain,
        "active_stage": project.active_stage,
        "completed_stages": list(project.completed_stages),
        "pdfs_count": len(project.parsed_papers),
        "discovered_works_count": len([w for w in project.discovered_works if not w.is_duplicate]),
        "prisma": prisma.model_dump(),
        "strongest_gaps": [
            f"{g.title} [{g.classification}, {g.supporting_publication}, {g.page_reference}]"
            for g in strongest_gaps[:5]
        ],
        "shared_methodologies": shared_methods,
        "methodology_justified": bool(project.methodology.proposed_architecture.strip() and project.research_gaps),
        "pending_experiments": [e.name for e in pending_exps],
        "measured_datasets": [d.filename for d in project.experiment_results.datasets],
        "objective_to_result_map": consistency.objective_to_result_map,
        "unsupported_claims": [f"[{c.support_status}] {c.claim_text[:90]}" for c in unsupported_claims[:6]],
        "manuscript_completion_pct": comp["completion_pct"],
        "approved_sections": comp["approved_count"],
        "consistency_score": consistency.overall_score,
        "open_consistency_issues": [i.description for i in consistency.issues if not i.resolved][:6],
    }


def _build_recommended_actions(diag: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if diag["pdfs_count"] == 0 and diag["discovered_works_count"] == 0:
        actions.append("Upload research PDFs in Workspace A (Feature 1) or search literature in Feature 4.")
    if diag["prisma"]["pending_screening"] > 0:
        actions.append(f"Screen {diag['prisma']['pending_screening']} pending paper(s) in Systematic Review (PRISMA).")
    if diag["pdfs_count"] > 0 and not diag["strongest_gaps"]:
        actions.append("Run Research Gap Detective to extract explicit and inferred gaps.")
    if not diag["methodology_justified"]:
        actions.append("Complete Stage 6 (Methodology & Architecture) and link it to your identified gaps.")
    if not diag["measured_datasets"]:
        actions.append("Upload real CSV/Excel experimental measurements in Stage 9 / Experiment Engine.")
    if diag["unsupported_claims"]:
        actions.append(f"Resolve {len(diag['unsupported_claims'])} unverified/contradicted manuscript claim(s).")
    if diag["open_consistency_issues"]:
        actions.append(f"Address {len(diag['open_consistency_issues'])} open consistency issue(s) in Stage 11.")
    if not actions:
        actions.append("Project checks look healthy — proceed to Stage 12 Publication Preparation & Export.")
    return actions


def ask_project_workflow_assistant(
    project: ResearchProject,
    question: str,
    vector_store: SessionVectorStore | None = None,
    llm_client: GeminiClient | None = None,
) -> dict[str, Any]:
    """
    Run a LangGraph contextual workflow query over the active project state and FAISS RAG store.
    Never modifies approved manuscript content.
    """
    builder = StateGraph(_AssistantState)

    def inspect_project_node(state: _AssistantState) -> _AssistantState:
        diag = compute_project_diagnostics(project)
        recs = _build_recommended_actions(diag)
        return {**state, "diagnostics": diag, "recommended_actions": recs}

    def retrieve_rag_node(state: _AssistantState) -> _AssistantState:
        hits_out: list[dict[str, Any]] = []
        if vector_store and vector_store.chunks and state.get("question"):
            hits = vector_store.search(state["question"], top_k=4, llm_client=llm_client)
            hits_out = [
                {
                    "paper": c.paper_title,
                    "pages": f"pp. {c.page_start}-{c.page_end}",
                    "section": c.section,
                    "excerpt": c.text[:350],
                }
                for c, _ in hits
            ]
        return {**state, "rag_hits": hits_out}

    def synthesize_answer_node(state: _AssistantState) -> _AssistantState:
        diag = state.get("diagnostics") or {}
        recs = state.get("recommended_actions") or []
        rag_hits = state.get("rag_hits") or []
        q = state.get("question", "").strip()

        if llm_client and llm_client.is_configured:
            try:
                approved_ctx = build_approved_context_summary(project)
                prompt = (
                    f"Answer the researcher's project-specific question strictly using the verified project diagnostics "
                    f"and retrieved PDF passages below.\n\n"
                    f"QUESTION: {q}\n\n"
                    f"APPROVED CONTEXT:\n{approved_ctx}\n\n"
                    f"LIVE DIAGNOSTICS:\n{diag}\n\n"
                    f"RETRIEVED PDF PASSAGES:\n{rag_hits or 'None'}\n\n"
                    f"Be concise, cite exact paper titles/pages when referencing PDFs, and state honestly if evidence is missing."
                )
                ans = llm_client.generate_text(prompt, system_instruction=SYSTEM_RESEARCH_ARCHITECT, temperature=0.2)
                return {**state, "answer": ans}
            except Exception:
                pass

        # Deterministic structured diagnostic answer based on the question keywords
        ql = q.lower()
        parts: list[str] = [f"**Project:** {diag.get('title')} (`{diag.get('domain')}` | Stage: `{diag.get('active_stage')}`)"]
        if "gap" in ql or "strongest" in ql:
            gaps = diag.get("strongest_gaps") or []
            parts.append("**Strongest Evidence Gaps:**\n" + ("\n".join(f"- {g}" for g in gaps) if gaps else "- No gaps extracted yet."))
        if "method" in ql or "similar" in ql or "justif" in ql:
            shared = diag.get("shared_methodologies") or {}
            parts.append("**Shared Methodologies Across Papers:** " + (str(shared) if shared else "Fewer than 2 papers share extracted algorithms so far."))
            parts.append(f"**Methodology Justified by Gaps & Architecture:** {'Yes' if diag.get('methodology_justified') else 'Not yet — complete Stage 6 and Gap Detective.'}")
        if "literature" in ql or "missing" in ql or "prisma" in ql:
            p = diag.get("prisma", {})
            parts.append(
                f"**Literature Review Status:** {p.get('studies_included', 0)} included, "
                f"{p.get('pending_screening', 0)} pending screening, {p.get('needs_review', 0)} needs review."
            )
        if "experiment" in ql or "pending" in ql or "result" in ql or "objective" in ql:
            parts.append(f"**Measured Datasets:** {diag.get('measured_datasets') or ['None uploaded']}")
            parts.append(f"**Pending Experiments:** {diag.get('pending_experiments') or ['None logged']}")
            parts.append(f"**Objective-to-Result Map:** {diag.get('objective_to_result_map') or ['No objectives defined yet']}")
        if "claim" in ql or "evidence" in ql or len(parts) == 1:
            unsup = diag.get("unsupported_claims") or []
            parts.append("**Claims Needing Evidence:**\n" + ("\n".join(f"- {u}" for u in unsup) if unsup else "- None flagged (run Claim Verifier after drafting)."))
            if rag_hits:
                parts.append("**Matching PDF Evidence:**\n" + "\n".join(f"- *{h['paper']}* ({h['pages']}): {h['excerpt'][:180]}..." for h in rag_hits[:3]))

        parts.append("**Recommended Next Actions:**\n" + "\n".join(f"- {r}" for r in recs))
        return {**state, "answer": "\n\n".join(parts)}

    builder.add_node("inspect_project", inspect_project_node)
    builder.add_node("retrieve_rag", retrieve_rag_node)
    builder.add_node("synthesize_answer", synthesize_answer_node)

    builder.add_edge(START, "inspect_project")
    builder.add_edge("inspect_project", "retrieve_rag")
    builder.add_edge("retrieve_rag", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)

    graph = builder.compile()
    result = graph.invoke({"question": question})
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "diagnostics": result.get("diagnostics", {}),
        "rag_hits": result.get("rag_hits", []),
        "recommended_actions": result.get("recommended_actions", []),
    }
