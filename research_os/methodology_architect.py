"""Part 5: Methodology Architect — grounded method, algorithm, dataset, and evaluation recommendations."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import MethodologyRecommendation, ResearchProject


class _MethRecPayload(BaseModel):
    recommendations: List[MethodologyRecommendation] = Field(default_factory=list)


def recommend_methodologies(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> list[MethodologyRecommendation]:
    """
    Recommend grounded algorithms, datasets, baselines, and evaluation protocols
    explaining suitability, problem addressed, assumptions, and limitations.
    """
    topic = project.selected_topic or project.title
    prob = project.problem_statement.problem_definition or topic
    lit_methods = []
    lit_datasets = []
    for a in project.paper_analyses.values():
        lit_methods.extend(x.name for x in a.algorithms[:3])
        lit_datasets.extend(d.name for d in a.datasets[:3])

    if llm_client and llm_client.is_configured:
        prompt = (
            f"As Methodology Architect, recommend 3 candidate methodologies for:\n"
            f"Topic: {topic}\nDomain: {project.domain}\nProblem: {prob}\n"
            f"Objectives: {project.problem_statement.objectives}\n"
            f"Algorithms seen in uploaded literature: {list(dict.fromkeys(lit_methods)) or 'None'}\n"
            f"Datasets seen in uploaded literature: {list(dict.fromkeys(lit_datasets)) or 'None'}\n\n"
            f"For every MethodologyRecommendation explain suitability_reason, problem_addressed, "
            f"assumptions, limitations, evaluation_plan, recommended_datasets, recommended_metrics, and baselines."
        )
        try:
            payload = llm_client.generate_structured(prompt, _MethRecPayload, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            if payload.recommendations:
                project.methodology_recommendations = payload.recommendations
                return payload.recommendations
        except Exception:
            pass

    ds_defaults = list(dict.fromkeys(lit_datasets))[:3] or [f"Public {project.domain} Benchmark Dataset", "Out-of-Distribution Validation Split"]
    recs = [
        MethodologyRecommendation(
            method_name="Modular Hybrid Architecture with Regularized Representation",
            suitability_reason=f"Balances strong task accuracy with interpretable module-level ablation for {topic}.",
            problem_addressed=prob[:220],
            assumptions=[
                "Clean train/validation/test separation without data leakage",
                "Stationary feature schema across baseline and proposed runs",
            ],
            limitations=[
                "Requires hyperparameter tuning on validation split",
                "Compute cost scales with representation dimensionality",
            ],
            evaluation_plan="Compare against classical and neural baselines across 3 random seeds plus component ablation.",
            recommended_datasets=ds_defaults,
            recommended_metrics=["Accuracy", "Macro-F1", "ROC-AUC", "Inference Latency (ms)"],
            baselines=["Majority / Heuristic Baseline", "Linear / Classical ML Baseline"] + (lit_methods[:2] or ["Published Neural Baseline"]),
        ),
        MethodologyRecommendation(
            method_name="Retrieval-Augmented / Evidence-Grounded Pipeline",
            suitability_reason=f"Directly targets factual grounding and traceability gaps in {project.domain}.",
            problem_addressed=f"Reduces unsupported errors and improves auditability for {topic}.",
            assumptions=[
                "Curated reference corpus is representative and indexed accurately",
                "Retrieval recall is sufficient at top-k",
            ],
            limitations=[
                "Sensitive to retriever top-k coverage and chunk boundaries",
                "Additional retrieval latency overhead",
            ],
            evaluation_plan="Measure retrieval Recall@k alongside downstream task accuracy and contradiction rate.",
            recommended_datasets=ds_defaults,
            recommended_metrics=["Recall@5", "Exact Match / F1", "Faithfulness / Support Rate"],
            baselines=["Zero-shot / Ungrounded Baseline", "BM25 Lexical Retrieval Baseline"],
        ),
    ]
    project.methodology_recommendations = recs
    return recs
