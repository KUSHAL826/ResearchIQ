"""Part 4: Research Problem Architect — literature-grounded candidate comparison & selection."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ProblemCandidate, ProblemStatementData, ResearchProject


class _CandidatePayload(BaseModel):
    candidates: List[ProblemCandidate] = Field(default_factory=list)


def generate_problem_candidates(
    domain: str,
    interest: str,
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> list[ProblemCandidate]:
    """
    Analyze uploaded literature, limitations, and gaps to generate 3 comparable
    Research Problem Candidates for structured evaluation and selection.
    """
    papers_ctx = [p.title for p in project.parsed_papers[:5]] + [
        w.title for w in project.discovered_works if not w.is_duplicate
    ][:5]
    lims_ctx: list[str] = []
    for a in project.paper_analyses.values():
        for lim in a.limitations[:3]:
            lims_ctx.append(f"{lim.description[:120]} ({a.paper_title[:30]}, {lim.page_reference})")
    gaps_ctx = [f"{g.title}: {g.description[:120]}" for g in project.research_gaps[:5]]

    if llm_client and llm_client.is_configured:
        prompt = (
            f"Act as Research Problem Architect for domain '{domain}' and student interest '{interest}'.\n"
            f"Uploaded/Discovered Literature: {papers_ctx or 'None yet'}\n"
            f"Extracted Limitations: {lims_ctx or 'None yet'}\n"
            f"Identified Gaps: {gaps_ctx or 'None yet'}\n\n"
            f"Generate 3 distinct, comparable ProblemCandidate objects with realistic feasibility_score (1-10) "
            f"and contribution_score (1-10), supporting_papers, existing_solutions, identified_limitations, "
            f"potential_contribution, 2-3 research_questions, 3 objectives, required_resources, "
            f"suggested_methodology, and evaluation_strategy."
        )
        try:
            payload = llm_client.generate_structured(prompt, _CandidatePayload, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            if payload.candidates:
                project.problem_candidates = payload.candidates
                return payload.candidates
        except Exception:
            pass

    focus = interest.strip() or f"{domain} Systems"
    sup = papers_ctx[:3] if papers_ctx else ["Requires literature search in Workspace A"]
    lims = lims_ctx[:3] if lims_ctx else ["Limited cross-dataset generalization", "Insufficient ablation transparency"]
    candidates = [
        ProblemCandidate(
            title=f"Robustness & Generalization Benchmark for {focus}",
            problem_statement=(
                f"Existing approaches to {focus} in {domain} are frequently evaluated on narrow single-domain splits, "
                f"making real-world robustness under distribution shift uncertain."
            ),
            supporting_papers=sup,
            existing_solutions=["Standard supervised baselines", "Single-benchmark fine-tuning"],
            identified_limitations=lims,
            potential_contribution=f"Quantifies failure boundaries and introduces a domain-adaptive evaluation framework for {focus}.",
            research_questions=[
                f"RQ1: How severely do existing {focus} models degrade under cross-dataset shift?",
                "RQ2: Which domain adaptation or calibration mechanisms best recover performance?",
            ],
            objectives=[
                f"O1: Audit baseline robustness across multiple {domain} datasets.",
                f"O2: Implement a calibrated adaptation module for {focus}.",
                "O3: Validate statistical gains via ablation and error analysis.",
            ],
            required_resources=["2+ public benchmark datasets", "Python / PyTorch or scikit-learn", "Baseline implementations"],
            suggested_methodology="Multi-dataset cross-evaluation with uncertainty calibration and controlled ablation.",
            evaluation_strategy="Accuracy, Macro-F1, Calibration Error / ROC-AUC across in-domain and out-of-domain splits.",
            feasibility_score=8.5,
            contribution_score=8.0,
        ),
        ProblemCandidate(
            title=f"Lightweight Efficiency-Aware Architecture for {focus}",
            problem_statement=(
                f"High-performing {domain} models for {focus} incur substantial latency and memory overhead, "
                f"hindering deployment in resource-constrained environments."
            ),
            supporting_papers=sup,
            existing_solutions=["Heavyweight neural ensembles", "Uncompressed transformer backbones"],
            identified_limitations=lims,
            potential_contribution=f"Delivers an accuracy-efficiency Pareto frontier for {focus} without sacrificing interpretability.",
            research_questions=[
                f"RQ1: What is the accuracy-latency trade-off across compact architectures for {focus}?",
                "RQ2: Which components can be pruned or distilled with minimal metric loss?",
            ],
            objectives=[
                "O1: Profile latency, memory, and accuracy of current heavyweight baselines.",
                f"O2: Design a compact hybrid pipeline tailored to {focus}.",
                "O3: Benchmark Pareto efficiency against baselines.",
            ],
            required_resources=["Benchmark dataset", "Runtime profiler", "Baseline model checkpoints"],
            suggested_methodology="Hybrid feature compression / distillation with strict compute-budgeted comparisons.",
            evaluation_strategy="F1-score / Accuracy vs inference latency (ms) and parameter count.",
            feasibility_score=8.0,
            contribution_score=7.8,
        ),
        ProblemCandidate(
            title=f"Evidence-Grounded & Explainable {focus} Framework",
            problem_statement=(
                f"Current {focus} systems provide opaque predictions without verifiable source attribution, "
                f"reducing trustworthiness in high-stakes {domain} workflows."
            ),
            supporting_papers=sup,
            existing_solutions=["Black-box classifiers", "Unverified end-to-end generation"],
            identified_limitations=lims,
            potential_contribution=f"Integrates verifiable evidence attribution and error auditing directly into {focus}.",
            research_questions=[
                f"RQ1: Does coupling evidence retrieval with prediction reduce unsupported errors in {focus}?",
                "RQ2: How do attribution checks affect overall task accuracy and human auditability?",
            ],
            objectives=[
                "O1: Measure unsupported output rate in ungrounded baselines.",
                f"O2: Build an evidence-verifying pipeline for {focus}.",
                "O3: Evaluate factual consistency and task metrics.",
            ],
            required_resources=["Annotated evaluation set", "Retrieval index (FAISS)", "Verification rubric"],
            suggested_methodology="Retrieval-grounded pipeline with explicit claim/prediction verification scoring.",
            evaluation_strategy="Task metric + Attribution Precision + Contradiction Rate.",
            feasibility_score=7.3,
            contribution_score=8.6,
        ),
    ]
    project.problem_candidates = candidates
    return candidates


def select_problem_candidate(project: ResearchProject, candidate: ProblemCandidate) -> None:
    """Apply a chosen ProblemCandidate into the project's active ProblemStatementData without auto-locking."""
    project.selected_topic = candidate.title
    project.title = candidate.title[:140]
    project.proposal.title = candidate.title
    project.problem_statement = ProblemStatementData(
        problem_definition=candidate.problem_statement,
        motivation=candidate.potential_contribution,
        objectives=list(candidate.objectives),
        scope_in=[candidate.evaluation_strategy] if candidate.evaluation_strategy else [],
        scope_out=["Unverified proprietary benchmarks outside evaluation scope"],
        research_questions=list(candidate.research_questions),
        hypotheses=[
            f"H1: {candidate.suggested_methodology} improves target metrics over existing solutions.",
        ],
        approved=False,
    )
