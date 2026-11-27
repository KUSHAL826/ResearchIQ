"""Studio Stage 7: Implementation Guidance, Starter Code (clearly labelled), and Checklist."""

from __future__ import annotations

from agents.master_orchestrator import build_approved_context_summary
from agents.prompts import SYSTEM_RESEARCH_ARCHITECT
from core.llm_client import GeminiClient
from core.schemas import ImplementationPlan, ResearchProject


STARTER_CODE_HEADER = (
    "# ============================================================================\n"
    "# RESEARCHIQ GENERATED STARTING POINT — REQUIRES TESTING AND VALIDATION\n"
    "# Verify dataset paths, hyperparameters, random seeds, and evaluation splits\n"
    "# before reporting any results in your manuscript.\n"
    "# ============================================================================\n\n"
)


def generate_implementation_plan(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> ImplementationPlan:
    """Generate structured implementation plan and clearly labelled starter code."""
    approved_ctx = build_approved_context_summary(project)
    if llm_client and llm_client.is_configured:
        prompt = (
            f"Generate Stage 7 ImplementationPlan for:\n{approved_ctx}\n"
            f"Methodology: {project.methodology.proposed_architecture}\n"
            f"Algorithms: {project.methodology.algorithms}\n"
            f"Include:\n"
            f"- tech_stack: list of dicts with 'layer', 'tool', 'rationale'\n"
            f"- modules: list of dicts with 'module', 'responsibility', 'depends_on'\n"
            f"- starter_code: clean Python starting-point template (no fake results)\n"
            f"- checklist: list of dicts with 'task', 'done' (bool)\n"
            f"Keep approved=False."
        )
        try:
            res = llm_client.generate_structured(prompt, ImplementationPlan, system_instruction=SYSTEM_RESEARCH_ARCHITECT)
            res.approved = False
            if res.starter_code and "STARTING POINT" not in res.starter_code:
                res.starter_code = STARTER_CODE_HEADER + res.starter_code
            return res
        except Exception:
            pass

    code_template = (
        STARTER_CODE_HEADER
        + "import random\nimport numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
        + "from sklearn.metrics import accuracy_score, f1_score\n\n\n"
        + "def set_seed(seed: int = 42) -> None:\n"
        + "    random.seed(seed)\n    np.random.seed(seed)\n\n\n"
        + "def run_experiment_template(csv_path: str, target_col: str) -> dict:\n"
        + "    set_seed(42)\n"
        + "    df = pd.read_csv(csv_path)\n"
        + "    X = df.drop(columns=[target_col])\n"
        + "    y = df[target_col]\n"
        + "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        + "    # TODO: Plug in your proposed model and baselines here\n"
        + "    raise NotImplementedError('Implement model training and validation on your actual dataset.')\n"
    )
    return ImplementationPlan(
        tech_stack=[
            {"layer": "Core Language", "tool": "Python 3.11+", "rationale": "Ecosystem support for ML & scientific computing"},
            {"layer": "Data & Metrics", "tool": "pandas / scikit-learn", "rationale": "Deterministic preprocessing and metric logging"},
            {"layer": "Experiment Tracking", "tool": "CSV / JSON Logs", "rationale": "Directly importable into RESEARCHIQ Stage 9"},
        ],
        preprocessing_steps=[
            "Validate schema and missing values",
            "Split train/validation/test prior to fitting scalers or encoders",
            "Log random seeds and preprocessing parameters",
        ],
        modules=[
            {"module": "data_loader.py", "responsibility": "Dataset ingestion & split validation", "depends_on": "None"},
            {"module": "model.py", "responsibility": "Proposed architecture & baseline wrappers", "depends_on": "data_loader.py"},
            {"module": "evaluate.py", "responsibility": "Metric computation & CSV export", "depends_on": "model.py"},
        ],
        starter_code=code_template,
        code_explanation="Modular skeleton separating deterministic data splitting, model execution, and metric logging. Starting point only — requires implementation and testing on real data.",
        documentation_notes="Record exact package versions, seeds, and dataset splits for manuscript reproducibility.",
        checklist=[
            {"task": "Verify dataset integrity and leakage-free splits", "done": False},
            {"task": "Implement baseline models", "done": False},
            {"task": "Implement proposed architecture", "done": False},
            {"task": "Run unit tests on preprocessing & metrics", "done": False},
            {"task": "Export experiment metrics CSV for Stage 9", "done": False},
        ],
        approved=False,
    )
