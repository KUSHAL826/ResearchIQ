"""Part 7: Advanced Manuscript Intelligence — Writing Assistant, Claim Mapping, Completeness, Reproducibility & Consistency."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from core.llm_client import GeminiClient
from core.schemas import (
    ClaimVerificationResult,
    ManuscriptSectionState,
    ReproducibilityCheckReport,
    ReproducibilityItem,
    ResearchProject,
)
from core.vector_store import SessionVectorStore
from paper_studio.manuscript_manager import check_reference_completeness
from research.citation_verifier import extract_claims_from_manuscript, verify_claim_against_evidence
from research.consistency_checker import run_consistency_check


def analyze_writing_quality(
    section_name: str,
    text: str,
    llm_client: GeminiClient | None = None,
) -> dict[str, Any]:
    """
    Analyze academic tone, repetition, unsupported claims, missing transitions,
    and terminology consistency without mutating approved sections.
    """
    clean = (text or "").strip()
    if not clean:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "repeated_words": [],
            "unsupported_statements": [],
            "tone_notes": ["Section is empty — draft content first."],
            "suggested_revision": "",
        }

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    words = re.findall(r"[a-z]{4,}", clean.lower())
    stop = {"that", "this", "with", "from", "have", "were", "been", "which", "their", "into", "also", "these", "using"}
    freq = Counter(w for w in words if w not in stop)
    repeated = [f"{w} ({c}x)" for w, c in freq.most_common(6) if c >= 4]

    # Flag strong empirical/comparative claims lacking citations or numbers
    strong_claim_re = re.compile(r"\b(outperform|proves|always|never|significant|state-of-the-art|novel|superior)\b", re.IGNORECASE)
    unsupported = [
        s[:140]
        for s in sentences
        if strong_claim_re.search(s) and not re.search(r"(\[\d+\]|\d+(?:\.\d+)?%?)", s)
    ]

    informal = [w for w in ("really", "very", "a lot", "kind of", "stuff", "things", "basically") if re.search(rf"\b{w}\b", clean, re.IGNORECASE)]
    tone_notes = []
    if informal:
        tone_notes.append(f"Informal phrasing detected: {', '.join(informal)}.")
    if len(sentences) >= 3:
        no_trans = sum(1 for s in sentences[1:] if s.split()[0].lower() in {"this", "it", "the"})
        if no_trans >= max(2, len(sentences) // 2):
            tone_notes.append("Consider adding explicit academic transitions (e.g., 'Consequently,', 'In contrast,', 'Specifically,').")
    if not tone_notes:
        tone_notes.append("Academic register and sentence flow look solid.")

    suggested_revision = ""
    if llm_client and llm_client.is_configured:
        try:
            prompt = (
                f"Improve the academic clarity, tone, transitions, and conciseness of this '{section_name}' draft.\n"
                f"STRICT RULE: Preserve the researcher's exact factual meaning, citations, and numeric values.\n\n"
                f"{clean[:6000]}"
            )
            suggested_revision = llm_client.generate_text(prompt, temperature=0.2)
        except Exception:
            suggested_revision = ""

    return {
        "word_count": len(clean.split()),
        "sentence_count": len(sentences),
        "repeated_words": repeated,
        "unsupported_statements": unsupported[:6],
        "tone_notes": tone_notes,
        "suggested_revision": suggested_revision,
    }


def map_claims_to_evidence(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient | None = None,
) -> list[ClaimVerificationResult]:
    """Map manuscript citation claims to indexed PDF passages and highlight unsupported ones."""
    extracted = extract_claims_from_manuscript(project.manuscript_sections)
    refs_sec = project.manuscript_sections.get("References")
    refs_text = refs_sec.content if refs_sec else ""
    mapped: list[ClaimVerificationResult] = []
    for sec, clm, marker in extracted:
        mapped.append(
            verify_claim_against_evidence(
                claim_text=clm,
                cited_source=marker,
                section_name=sec,
                vector_store=vector_store,
                discovered_works=project.discovered_works,
                llm_client=llm_client,
                parsed_papers=project.parsed_papers,
                references_text=refs_text,
            )
        )
    if mapped:
        project.claim_verifications = mapped
    return mapped


def analyze_manuscript_completeness(project: ResearchProject) -> list[dict[str, Any]]:
    """Evaluate 10 core academic completeness dimensions strictly from actual project state."""
    ms = project.manuscript_sections
    sec_text = lambda k: (ms.get(k).content.strip() if k in ms and ms.get(k) else "")
    ref_chk = check_reference_completeness(project)
    consistency = run_consistency_check(project)

    checks = [
        ("Problem clarity", bool(project.problem_statement.problem_definition.strip()) or len(sec_text("Introduction")) > 80),
        ("Research motivation", bool(project.problem_statement.motivation.strip()) or "motivat" in sec_text("Introduction").lower()),
        ("Literature coverage", len([w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]) > 0 or len(sec_text("Literature Review")) > 80),
        ("Gap justification", len(project.research_gaps) > 0 or len(sec_text("Research Gap")) > 60),
        ("Objective alignment", len(project.problem_statement.objectives) > 0 and len(sec_text("Objectives")) > 30),
        ("Methodology clarity", bool(project.methodology.proposed_architecture.strip()) or len(sec_text("Proposed Methodology")) > 80),
        ("Experimental completeness", len(project.experiment_design.evaluation_metrics) > 0 and len(sec_text("Experimental Setup")) > 40),
        ("Result interpretation", bool(project.experiment_results.datasets) and len(sec_text("Results and Discussion")) > 60),
        ("Conclusion consistency", len(sec_text("Conclusion")) > 50 and consistency.overall_score >= 60),
        ("Reference completeness", ref_chk["is_complete"]),
    ]

    rows = []
    for dim, ok in checks:
        rows.append({"Dimension": dim, "Status": "Complete" if ok else "Needs Attention", "Score": 100 if ok else 30})
    return rows


def check_research_reproducibility(project: ResearchProject) -> ReproducibilityCheckReport:
    """
    Audit whether the manuscript and studio stages report dataset, preprocessing,
    model config, hyperparameters, experimental setup, metrics, hardware/software, and seeds.
    Never invents missing details.
    """
    blob = "\n".join(
        [
            project.methodology.dataset_requirements,
            project.methodology.proposed_architecture,
            " ".join(project.implementation.preprocessing_steps),
            project.implementation.documentation_notes,
            project.implementation.starter_code,
            " ".join(f"{c.get('parameter', '')} {c.get('values', '')}" for c in project.experiment_design.configurations),
            " ".join(project.experiment_design.evaluation_metrics),
            *(sec.content for sec in project.manuscript_sections.values() if sec.content),
        ]
    ).lower()

    criteria = [
        ("Dataset", ("dataset", "benchmark", "split", "corpus", ".csv", "rows"), "Specify dataset name, source, and train/val/test split sizes."),
        ("Preprocessing", ("preprocess", "normaliz", "clean", "tokeniz", "scaling", "imput"), "Document data cleaning, filtering, and feature transforms."),
        ("Model configuration", ("architecture", "layer", "encoder", "transformer", "model", "module"), "Detail model architecture, layers, and baselines."),
        ("Hyperparameters", ("learning rate", "epoch", "batch", "alpha", "lambda", "hyperparameter", "depth"), "Report final hyperparameters and tuning protocol."),
        ("Experimental setup", ("baseline", "ablation", "cross-validation", "protocol", "configuration"), "Describe baseline settings and ablation protocol."),
        ("Evaluation metrics", ("accuracy", "f1", "precision", "recall", "auc", "rmse", "latency"), "State exact metric definitions and averaging mode."),
        ("Hardware/software environment", ("gpu", "cpu", "ram", "python", "pytorch", "tensorflow", "scikit", "cuda"), "Record OS, Python/library versions, and CPU/GPU specs."),
        ("Random seeds", ("seed", "random_state", "deterministic", "reproduc"), "Report random seeds and variance across multiple runs."),
    ]

    items: list[ReproducibilityItem] = []
    reported_cnt = 0
    for name, kws, rec in criteria:
        hits = [kw for kw in kws if kw in blob]
        if len(hits) >= 2:
            status = "Reported"
            reported_cnt += 1
            ev = f"Matched keywords in manuscript/studio: {', '.join(hits[:4])}"
        elif len(hits) == 1:
            status = "Partial"
            reported_cnt += 0.5  # type: ignore[assignment]
            ev = f"Partial mention ({hits[0]}); verify full details are explicitly stated."
        else:
            status = "Missing"
            ev = "Not detected in current manuscript or studio configurations."
        items.append(ReproducibilityItem(criterion=name, status=status, evidence=ev, recommendation=rec if status != "Reported" else "Reported."))

    score = round((reported_cnt / len(criteria)) * 100.0, 1)
    report = ReproducibilityCheckReport(items=items, score=score)
    project.reproducibility_report = report
    return report


def run_academic_consistency_intelligence(project: ResearchProject) -> dict[str, Any]:
    """Cross-check RQs -> Objectives -> Methodology -> Experiments -> Results -> Conclusions non-destructively."""
    base_report = run_consistency_check(project)
    rqs = project.problem_statement.research_questions
    objs = project.problem_statement.objectives
    rq_to_obj = []
    obj_blob = " ".join(objs).lower()
    for idx, rq in enumerate(rqs, start=1):
        kws = [w for w in re.findall(r"[a-z]{4,}", rq.lower()) if w not in {"what", "which", "does", "with", "that", "from"}]
        linked = any(k in obj_blob for k in kws) if kws and objs else False
        rq_to_obj.append({"Research Question": f"RQ{idx}: {rq[:75]}", "Aligned to Objectives": "Yes" if linked else "Needs link"})

    return {
        "overall_score": base_report.overall_score,
        "rq_to_objective_map": rq_to_obj,
        "objective_to_result_map": base_report.objective_to_result_map,
        "methodology_to_experiment_map": base_report.methodology_to_experiment_map,
        "issues": base_report.issues,
    }
