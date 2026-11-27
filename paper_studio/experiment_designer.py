"""Studio Stage 8: Experiment Design, Baselines, Ablation Plan & Resource Estimates."""

from __future__ import annotations

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ExperimentDesign, ResearchProject


def generate_experiment_design(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> ExperimentDesign:
    """Design baselines, configurations, metrics, ablation plan, and testing procedures."""
    approved_ctx = build_approved_context_summary(project)
    if llm_client and llm_client.is_configured:
        prompt = (
            f"Create Stage 8 ExperimentDesign for:\n{approved_ctx}\n"
            f"Methodology: {project.methodology.proposed_architecture}\n"
            f"Algorithms: {project.methodology.algorithms}\n"
            f"Include:\n"
            f"- baselines: list of dicts with 'name', 'type', 'rationale'\n"
            f"- configurations: list of dicts with 'parameter', 'values', 'notes'\n"
            f"- evaluation_metrics: list of metric names\n"
            f"- ablation_plan: list of dicts with 'variant', 'removed_or_changed', 'hypothesis'\n"
            f"- testing_procedures: list of protocol steps\n"
            f"- resource_estimates: list of dicts with 'stage', 'estimated_hours', 'compute'\n"
            f"Keep approved=False."
        )
        try:
            res = llm_client.generate_structured(prompt, ExperimentDesign, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            res.approved = False
            return res
        except Exception:
            pass

    return ExperimentDesign(
        baselines=[
            {"name": "Simple Heuristic / Majority Baseline", "type": "Lower Bound", "rationale": "Establishes task floor"},
            {"name": "Standard Classical Baseline", "type": "Traditional ML", "rationale": "Tests non-deep representation value"},
            {"name": "Published SOTA Reference", "type": "Literature Baseline", "rationale": "Benchmarks against prior work"},
        ],
        configurations=[
            {"parameter": "Random Seeds", "values": "42, 123, 456", "notes": "Report mean +/- std across 3 runs"},
            {"parameter": "Train/Val/Test Split", "values": "70% / 15% / 15%", "notes": "Stratified & leakage-free"},
            {"parameter": "Learning Rate / Regularization", "values": "Tuned on validation split only", "notes": "No test leakage"},
        ],
        evaluation_metrics=["Accuracy", "Precision", "Recall", "F1-Score", "Runtime (s)"],
        ablation_plan=[
            {"variant": "Full Proposed Model", "removed_or_changed": "None", "hypothesis": "Best overall metric balance"},
            {"variant": "w/o Feature Encoder", "removed_or_changed": "Replace encoder with raw features", "hypothesis": "Measures representation gain"},
            {"variant": "w/o Regularization / Constraint", "removed_or_changed": "Disable constraint term", "hypothesis": "Measures generalization stability"},
        ],
        testing_procedures=[
            "Verify deterministic preprocessing and split hashes before training.",
            "Run all baselines under identical data splits and hardware budget.",
            "Export per-run metrics to CSV/Excel for Stage 9 automated analysis.",
        ],
        resource_estimates=[
            {"stage": "Data Prep", "estimated_hours": 4, "compute": "CPU"},
            {"stage": "Baseline Runs", "estimated_hours": 6, "compute": "CPU / 1x GPU"},
            {"stage": "Proposed + Ablations", "estimated_hours": 10, "compute": "1x GPU"},
        ],
        approved=False,
    )
