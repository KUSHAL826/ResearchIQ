"""Studio Stage 6: Methodology, Architecture, Algorithms & Interactive Mermaid Diagrams."""

from __future__ import annotations

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import MethodologyDesign, ResearchProject


DEFAULT_ARCH_MERMAID = """flowchart LR
    RAW["Raw Input Data"] --> PREP["Preprocessing & Validation"]
    PREP --> FEAT["Feature / Representation Encoder"]
    FEAT --> CORE["Core Proposed Module"]
    CORE --> EVAL["Evaluation & Ablation Engine"]
    EVAL --> OUT["Verified Metrics & Analysis"]"""

DEFAULT_WORKFLOW_MERMAID = """flowchart TD
    S1["1. Problem & Objectives"] --> S2["2. Dataset Curation"]
    S2 --> S3["3. Baseline Training"]
    S3 --> S4["4. Proposed Model Training"]
    S4 --> S5["5. Statistical & Ablation Audit"]"""

DEFAULT_PIPELINE_MERMAID = """flowchart LR
    D1[("Source Dataset")] --> D2["Cleaning & Split"]
    D2 --> D3["Train / Val / Test"]
    D3 --> D4["Batch Loader"]
    D4 --> D5["Model Inference"]
    D5 --> D6[("Metrics CSV")]"""


def generate_methodology_design(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> MethodologyDesign:
    """Generate research design, algorithms, mathematical framing, and Mermaid diagrams."""
    approved_ctx = build_approved_context_summary(project)
    if llm_client and llm_client.is_configured:
        prompt = (
            f"Design Stage 6 MethodologyDesign for:\n{approved_ctx}\n\n"
            f"Problem Statement: {project.problem_statement.problem_definition}\n"
            f"Proposal Summary: {project.proposal.methodology_summary}\n"
            f"Include:\n"
            f"- algorithms: list of dicts with keys 'name', 'role', 'complexity'\n"
            f"- mathematical_models: concise LaTeX/text formulation\n"
            f"- evaluation_methods: list of metric/protocol strings\n"
            f"- Valid Mermaid.js 'flowchart' syntax in mermaid_architecture, mermaid_workflow, and mermaid_data_pipeline.\n"
            f"Keep approved=False."
        )
        try:
            res = llm_client.generate_structured(prompt, MethodologyDesign, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            res.approved = False
            if not res.mermaid_architecture:
                res.mermaid_architecture = DEFAULT_ARCH_MERMAID
            if not res.mermaid_workflow:
                res.mermaid_workflow = DEFAULT_WORKFLOW_MERMAID
            if not res.mermaid_data_pipeline:
                res.mermaid_data_pipeline = DEFAULT_PIPELINE_MERMAID
            return res
        except Exception:
            pass

    return MethodologyDesign(
        research_design="Comparative empirical research design combining controlled baseline benchmarking with component ablation.",
        proposed_architecture=(
            f"Modular pipeline for {project.selected_topic or project.title}: "
            "1) Schema-validated ingestion & preprocessing, 2) Representation encoder, "
            "3) Core task reasoning/prediction module, 4) Quantitative evaluator."
        ),
        data_flow_description="Input records pass through deterministic normalization and stratified train/val/test splitting prior to model training and metric logging.",
        algorithms=[
            {"name": "Preprocessing Normalizer", "role": "Outlier handling and feature scaling", "complexity": "O(N)"},
            {"name": "Proposed Core Model", "role": "Primary representation and prediction", "complexity": "Task-dependent"},
            {"name": "Ablation Comparator", "role": "Isolates each module contribution", "complexity": "O(K * E)"},
        ],
        mathematical_models="Objective: minimize L_total = L_task(y, y_hat) + lambda * R(theta) subject to validation constraints.",
        dataset_requirements="Reproducible public or curated domain dataset with explicit train/validation/test splits and clear label schema.",
        evaluation_methods=["Accuracy", "Macro F1-Score", "Precision / Recall", "Runtime (ms/sample)", "Ablation Delta"],
        mermaid_architecture=DEFAULT_ARCH_MERMAID,
        mermaid_workflow=DEFAULT_WORKFLOW_MERMAID,
        mermaid_data_pipeline=DEFAULT_PIPELINE_MERMAID,
        approved=False,
    )
