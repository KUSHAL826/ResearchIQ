"""Studio Stages 1 & 2: Research Topic Generator and Problem Statement Formulation."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ProblemStatementData, ResearchProject, TopicIdea


class TopicListPayload(BaseModel):
    topics: List[TopicIdea] = Field(default_factory=list)


class TitleImprovementPayload(BaseModel):
    critique: str = ""
    needs_improvement: bool = True
    improved_titles: List[str] = Field(default_factory=list)


def suggest_improved_titles(
    user_title: str,
    domain: str,
    secondary_domains: list[str] | None = None,
    llm_client: GeminiClient | None = None,
) -> tuple[str, list[str]]:
    """Evaluate a student's proposed paper title and suggest refined academic alternatives."""
    clean_title = user_title.strip()
    if not clean_title:
        return "Enter a working title first to evaluate specificity and academic framing.", []

    domains_str = ", ".join([domain] + [d for d in (secondary_domains or []) if d])
    if llm_client and llm_client.is_configured:
        prompt = (
            f"Evaluate this student's proposed research paper title in the domain '{domains_str}':\n"
            f"Proposed Title: \"{clean_title}\"\n\n"
            f"Provide:\n"
            f"1) critique: 1-2 crisp sentences evaluating clarity, technical specificity, and academic suitability.\n"
            f"2) needs_improvement: boolean indicating if refinement would strengthen the title.\n"
            f"3) improved_titles: exactly 3 distinct, publication-grade academic title improvements preserving the user's core topic."
        )
        try:
            payload = llm_client.generate_structured(
                prompt,
                TitleImprovementPayload,
                system_instruction=SYSTEM_RESEARCH_ARCHITECT,
            )
            suggestions = [t.strip() for t in payload.improved_titles if t.strip() and t.strip().lower() != clean_title.lower()]
            if suggestions:
                return payload.critique or "Evaluated title clarity and academic specificity.", suggestions[:3]
        except Exception:
            pass

    # Deterministic academic refinement when offline
    words = clean_title.split()
    if len(words) < 5:
        critique = (
            f"'{clean_title}' is concise ({len(words)} words) and works as a broad topic, "
            f"but adding method, evaluation focus, or domain scope makes it stronger for {domain} venues."
        )
    else:
        critique = (
            f"'{clean_title}' captures your core direction well. Below are sharper academic phrasings "
            f"highlighting methodology and empirical validation—or you may retain your original title."
        )
    suggestions = [
        f"{clean_title}: A Systematic Framework and Empirical Evaluation in {domain}",
        f"Evidence-Guided {clean_title} for Reliable {domain} Applications",
        f"Design and Quantitative Benchmarking of {clean_title}",
    ]
    return critique, suggestions


def generate_research_topics(
    domain: str,
    secondary_domains: list[str],
    seed_interest: str,
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> list[TopicIdea]:
    """Generate structured research topic ideas grounded in session papers and gaps."""
    gap_ctx = "; ".join(f"{g.title} ({g.supporting_publication})" for g in project.research_gaps[:6])
    lit_ctx = "; ".join(w.title for w in project.discovered_works[:6] if not w.is_duplicate)
    domains_str = ", ".join([domain] + [d for d in secondary_domains if d])

    if llm_client and llm_client.is_configured:
        prompt = (
            f"Generate 4 focused academic research topic ideas in: {domains_str}.\n"
            f"Student Seed Interest: {seed_interest or 'General exploration'}\n"
            f"Identified Session Gaps: {gap_ctx or 'None yet'}\n"
            f"Discovered Literature: {lit_ctx or 'None yet'}\n"
            f"Each TopicIdea must include 2-3 crisp research_questions, rationale, novelty_level, "
            f"feasibility_score (1-10), and impact_score (1-10)."
        )
        try:
            payload = llm_client.generate_structured(prompt, TopicListPayload, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            if payload.topics:
                return payload.topics
        except Exception:
            pass

    focus = seed_interest.strip() or f"{domain} Optimization"
    return [
        TopicIdea(
            title=f"Evidence-Grounded {focus} with Uncertainty-Aware Validation",
            domain=domain,
            research_questions=[
                f"RQ1: How do existing {domain} pipelines perform under domain shift?",
                f"RQ2: Does incorporating structured evidence verification improve reliability on {focus}?",
            ],
            rationale="Targets reliability and empirical reproducibility gaps commonly seen in single-benchmark studies.",
            novelty_level="Moderate-High",
            feasibility_score=8.2,
            impact_score=8.0,
        ),
        TopicIdea(
            title=f"Lightweight Hybrid Architecture for {focus} in Resource-Constrained Settings",
            domain=domain,
            research_questions=[
                f"RQ1: What accuracy-latency trade-offs emerge across compact vs heavyweight models in {focus}?",
                "RQ2: Which architectural modules contribute most under systematic ablation?",
            ],
            rationale="Balances practical deployment constraints with rigorous ablation analysis.",
            novelty_level="Moderate",
            feasibility_score=8.7,
            impact_score=7.6,
        ),
        TopicIdea(
            title=f"Cross-Domain Benchmark and Robustness Analysis for {domains_str}",
            domain=domains_str,
            research_questions=[
                f"RQ1: How well do models trained in {domain} generalize across {secondary_domains[0] if secondary_domains else 'adjacent tasks'}?",
                "RQ2: Which failure modes persist across datasets?",
            ],
            rationale="Directly addresses multi-dataset generalization and failure-mode transparency.",
            novelty_level="High",
            feasibility_score=7.4,
            impact_score=8.5,
        ),
    ]


def generate_problem_statement(
    topic: str,
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> ProblemStatementData:
    """Draft structured Problem Statement, Objectives, Scope, Research Questions, and Hypotheses."""
    gap_ctx = "\n".join(f"- {g.title}: {g.description} ({g.supporting_publication})" for g in project.research_gaps[:5])
    if llm_client and llm_client.is_configured:
        prompt = (
            f"Formulate a complete Stage 2 ProblemStatementData for the selected topic:\n"
            f"Topic: {topic}\nDomain: {project.domain}\n"
            f"Session Gaps:\n{gap_ctx or 'None uploaded yet'}\n"
            f"Include clear problem_definition, motivation, 3-4 measurable objectives, scope_in, scope_out, "
            f"3 research_questions, and 2 testable hypotheses. Keep approved=False so the student can review and approve."
        )
        try:
            res = llm_client.generate_structured(prompt, ProblemStatementData, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            res.approved = False
            return res
        except Exception:
            pass

    return ProblemStatementData(
        problem_definition=(
            f"Current approaches to '{topic}' in {project.domain} exhibit limited cross-dataset generalization, "
            f"insufficient component-level ablation clarity, and high sensitivity to noisy real-world inputs."
        ),
        motivation=(
            f"Addressing these limitations is essential for trustworthy academic and practical adoption of {project.domain} systems."
        ),
        objectives=[
            f"O1: Analyze baseline methods and quantify existing limitations in {topic}.",
            f"O2: Design a modular, reproducible methodology addressing the identified gaps.",
            f"O3: Conduct controlled experiments and ablation studies on benchmark datasets.",
            f"O4: Validate statistical performance, runtime efficiency, and failure boundaries.",
        ],
        scope_in=[
            f"Controlled benchmark evaluation within {project.domain}",
            "Quantitative comparison against reproducible baselines",
            "Component ablation and error analysis",
        ],
        scope_out=[
            "Unverified proprietary datasets without reproducible splits",
            "Hardware-specific kernel optimization outside experimental scope",
        ],
        research_questions=[
            f"RQ1: Which methodological bottlenecks limit baseline performance in {topic}?",
            "RQ2: To what extent does the proposed design improve target evaluation metrics over baselines?",
            "RQ3: What are the computational and generalization trade-offs across configurations?",
        ],
        hypotheses=[
            "H1: Integrating structured domain constraints will yield measurable improvements in primary evaluation metrics.",
            "H2: Systematic ablation will demonstrate non-redundant contributions from each core module.",
        ],
        approved=False,
    )
