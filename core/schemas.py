"""Pydantic v2 data models for RESEARCHIQ research state, RAG chunks, analyses, and .riq projects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Document Engine & RAG Models
# ---------------------------------------------------------------------------

class PageContent(BaseModel):
    page_number: int
    text: str
    char_count: int = 0
    detected_headings: list[str] = Field(default_factory=list)


class SectionBlock(BaseModel):
    canonical_name: str  # Abstract, Introduction, Literature Review, Methodology, Experiments, Results, Conclusion, Other
    raw_heading: str
    start_page: int
    end_page: int
    content: str
    summary: str = ""


class ExtractedTable(BaseModel):
    table_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    page_number: int
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class ExtractedFigure(BaseModel):
    figure_id: str = Field(default_factory=lambda: uuid4().hex[:8])
    page_number: int
    kind: Literal["figure", "equation", "diagram"] = "figure"
    caption: str = ""
    context_snippet: str = ""


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: uuid4().hex[:10])
    paper_id: str
    paper_title: str
    section: str
    page_start: int
    page_end: int
    text: str
    token_estimate: int = 0


class ParsedPaper(BaseModel):
    paper_id: str = Field(default_factory=lambda: f"paper_{uuid4().hex[:8]}")
    filename: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    num_pages: int = 0
    file_hash: str = ""
    asset_relpath: str = ""
    pages: list[PageContent] = Field(default_factory=list)
    sections: list[SectionBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    hierarchical_summary: str = ""
    processing_stages_completed: list[str] = Field(default_factory=list)
    # Explicit PRISMA screening fields for uploaded PDFs (no auto-inclusion)
    screening_status: Literal["unscreened", "included", "excluded", "needs_review"] = "unscreened"
    exclusion_reason: str = ""
    full_text_assessed: bool = False
    is_duplicate: bool = False


# ---------------------------------------------------------------------------
# Feature 1: Paper Intelligence
# ---------------------------------------------------------------------------

class AlgorithmItem(BaseModel):
    name: str
    role: str = ""
    inputs: str = ""
    outputs: str = ""
    page_reference: str = ""


class DatasetItem(BaseModel):
    name: str
    domain: str = ""
    size_or_split: str = ""
    metrics_used: list[str] = Field(default_factory=list)
    page_reference: str = ""


class MetricResultItem(BaseModel):
    metric: str
    value: float | None = None
    raw_value: str = ""
    model_or_method: str = ""
    dataset: str = ""
    page_reference: str = ""


class LimitationItem(BaseModel):
    description: str
    severity: Literal["High", "Medium", "Low"] = "Medium"
    page_reference: str = ""
    source_quote: str = ""


class PaperAnalysis(BaseModel):
    paper_id: str
    paper_title: str
    problem_statement: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    methodology_steps: list[str] = Field(default_factory=list)
    algorithms: list[AlgorithmItem] = Field(default_factory=list)
    datasets: list[DatasetItem] = Field(default_factory=list)
    results: list[MetricResultItem] = Field(default_factory=list)
    limitations: list[LimitationItem] = Field(default_factory=list)
    mermaid_methodology: str = ""
    mermaid_algorithms: str = ""
    analyzed_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Feature 2: Research Gap Detective
# ---------------------------------------------------------------------------

class ResearchGap(BaseModel):
    gap_id: str = Field(default_factory=lambda: f"gap_{uuid4().hex[:8]}")
    title: str
    description: str
    supporting_publication: str
    paper_id: str = ""
    page_reference: str = "N/A"
    classification: Literal["explicit", "inferred"] = "explicit"
    category: Literal["Methodological", "Dataset", "Evaluation", "Scalability", "Theoretical", "Domain"] = "Methodological"
    technical_significance: str = ""
    potential_improvement: str = ""
    feasibility_score: float = Field(default=6.0, ge=1.0, le=10.0)
    impact_score: float = Field(default=7.0, ge=1.0, le=10.0)
    implementation_difficulty: Literal["Low", "Medium", "High"] = "Medium"
    evidence_quote: str = ""


# ---------------------------------------------------------------------------
# Feature 3: Multi-Paper Intelligence
# ---------------------------------------------------------------------------

class ContradictionItem(BaseModel):
    topic: str
    paper_a: str
    claim_a: str
    paper_b: str
    claim_b: str
    analysis: str = ""


class MultiPaperComparison(BaseModel):
    selected_paper_ids: list[str] = Field(default_factory=list)
    common_themes: list[str] = Field(default_factory=list)
    methodology_matrix: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    gap_overlaps: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_summary: str = ""


# ---------------------------------------------------------------------------
# Feature 4 & 5: Scholarly Discovery & Systematic Literature Review
# ---------------------------------------------------------------------------

class ScholarlyWork(BaseModel):
    work_id: str
    source: Literal["openalex", "arxiv", "semantic_scholar", "uploaded"] = "openalex"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    citation_count: int | None = None
    fields_of_study: list[str] = Field(default_factory=list)
    referenced_ids: list[str] = Field(default_factory=list)
    cited_by_ids: list[str] = Field(default_factory=list)
    # Systematic Review screening fields
    screening_status: Literal["unscreened", "included", "excluded", "needs_review"] = "unscreened"
    exclusion_reason: str = ""
    screening_notes: str = ""
    full_text_assessed: bool = False
    is_duplicate: bool = False
    # Structured extraction for Literature Matrix
    extracted_problem: str = ""
    extracted_methodology: str = ""
    extracted_dataset: str = ""
    extracted_results: str = ""
    extracted_limitations: str = ""
    extracted_contribution: str = ""


class PRISMACounts(BaseModel):
    records_identified: int = 0
    duplicates_removed: int = 0
    records_screened: int = 0
    records_excluded: int = 0
    full_text_assessed: int = 0
    full_text_excluded: int = 0
    studies_included: int = 0
    needs_review: int = 0
    pending_screening: int = 0
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature 7: Cross-Domain Research
# ---------------------------------------------------------------------------

class CrossDomainIdea(BaseModel):
    idea_id: str = Field(default_factory=lambda: f"cd_{uuid4().hex[:6]}")
    title: str
    domains: list[str] = Field(default_factory=list)
    description: str
    supporting_works: list[str] = Field(default_factory=list)
    novelty_evidence: str = ""
    novelty_score: float = Field(default=6.0, ge=1.0, le=10.0)
    feasibility_score: float = Field(default=6.0, ge=1.0, le=10.0)
    impact_score: float = Field(default=7.0, ge=1.0, le=10.0)


# ---------------------------------------------------------------------------
# Feature 9 & 10: Citation Claim Verifier & Consistency Checker
# ---------------------------------------------------------------------------

class ClaimVerificationResult(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"clm_{uuid4().hex[:6]}")
    section_name: str = ""
    claim_text: str
    cited_source: str = ""
    support_status: Literal["Supported", "Partially supported", "Not supported", "Unable to verify"] = "Unable to verify"
    evidence_passage: str = ""
    page_reference: str = ""
    contradiction_note: str = ""
    reasoning: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_label: str = "Indicator only — not proof"


class ConsistencyIssue(BaseModel):
    issue_id: str = Field(default_factory=lambda: f"iss_{uuid4().hex[:6]}")
    category: Literal[
        "Missing information",
        "Contradictory statement",
        "Unsupported claim",
        "Inconsistent terminology",
        "Missing objective",
        "Unexplained result",
    ]
    severity: Literal["High", "Medium", "Low"] = "Medium"
    source_section: str = ""
    target_section: str = ""
    description: str = ""
    recommendation: str = ""
    resolved: bool = False


class ConsistencyReport(BaseModel):
    objective_to_result_map: list[dict[str, Any]] = Field(default_factory=list)
    methodology_to_experiment_map: list[dict[str, Any]] = Field(default_factory=list)
    claim_links: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[ConsistencyIssue] = Field(default_factory=list)
    overall_score: float = 0.0
    generated_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Workspace B: Research Paper Studio Stage Models
# ---------------------------------------------------------------------------

class TopicIdea(BaseModel):
    title: str
    domain: str
    research_questions: list[str] = Field(default_factory=list)
    rationale: str = ""
    novelty_level: str = "Moderate"
    feasibility_score: float = 7.0
    impact_score: float = 7.5


class ProblemStatementData(BaseModel):
    problem_definition: str = ""
    motivation: str = ""
    objectives: list[str] = Field(default_factory=list)
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    approved: bool = False


class ProposalData(BaseModel):
    title: str = ""
    background: str = ""
    problem_statement: str = ""
    motivation: str = ""
    objectives: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    methodology_summary: str = ""
    expected_contributions: list[str] = Field(default_factory=list)
    scope: str = ""
    milestones: list[dict[str, str]] = Field(default_factory=list)
    approved: bool = False


class MethodologyDesign(BaseModel):
    research_design: str = ""
    proposed_architecture: str = ""
    data_flow_description: str = ""
    algorithms: list[dict[str, str]] = Field(default_factory=list)
    mathematical_models: str = ""
    dataset_requirements: str = ""
    evaluation_methods: list[str] = Field(default_factory=list)
    mermaid_architecture: str = ""
    mermaid_workflow: str = ""
    mermaid_data_pipeline: str = ""
    approved: bool = False


class ImplementationPlan(BaseModel):
    tech_stack: list[dict[str, str]] = Field(default_factory=list)
    preprocessing_steps: list[str] = Field(default_factory=list)
    modules: list[dict[str, str]] = Field(default_factory=list)
    starter_code: str = ""
    code_explanation: str = ""
    documentation_notes: str = ""
    checklist: list[dict[str, Any]] = Field(default_factory=list)
    approved: bool = False


class ExperimentDesign(BaseModel):
    baselines: list[dict[str, str]] = Field(default_factory=list)
    configurations: list[dict[str, str]] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    ablation_plan: list[dict[str, str]] = Field(default_factory=list)
    testing_procedures: list[str] = Field(default_factory=list)
    resource_estimates: list[dict[str, Any]] = Field(default_factory=list)
    approved: bool = False


class UploadedDatasetInfo(BaseModel):
    dataset_id: str = Field(default_factory=lambda: f"ds_{uuid4().hex[:6]}")
    filename: str
    asset_relpath: str = ""
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)
    numeric_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentResultSummary(BaseModel):
    datasets: list[UploadedDatasetInfo] = Field(default_factory=list)
    selected_model_col: str = ""
    selected_metric_cols: list[str] = Field(default_factory=list)
    computed_classification_metrics: dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    confusion_labels: list[str] = Field(default_factory=list)
    discussion_notes: str = ""
    approved: bool = False


class ManuscriptSectionState(BaseModel):
    name: str
    content: str = ""
    approved: bool = False
    word_count: int = 0
    citations_used: list[str] = Field(default_factory=list)
    ai_notes: str = ""
    last_updated: str = Field(default_factory=utc_now_iso)


class PublicationPrepState(BaseModel):
    target_format: Literal["IEEE", "Springer", "ACM"] = "IEEE"
    authors: list[str] = Field(default_factory=lambda: ["First Author", "Second Author"])
    affiliation: str = "Department of Computer Science"
    keywords: list[str] = Field(default_factory=list)
    journal_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    guideline_checklist: list[dict[str, Any]] = Field(default_factory=list)
    readiness_score: float = 0.0


# ---------------------------------------------------------------------------
# RESEARCHIQ 2.0 — Advanced Intelligence Layer Schemas (Additive & Backward-Compatible)
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:6]}")
    statement: str
    source_paper: str = ""
    paper_id: str = ""
    page_number: str = "N/A"
    section: str = ""
    passage: str = ""
    retrieval_score: float = 0.0
    classification: Literal[
        "Direct evidence",
        "Supporting evidence",
        "Contradictory evidence",
        "Indirect evidence",
        "Insufficient evidence",
    ] = "Supporting evidence"


class ContradictionAnalysisItem(BaseModel):
    topic: str
    paper_a: str
    claim_a: str
    context_a: str = ""
    paper_b: str
    claim_b: str
    context_b: str = ""
    contradiction_type: Literal[
        "Conflicting results",
        "Experimental conclusion",
        "Methodology assumption",
        "Dataset-dependent",
        "Author claim",
    ] = "Dataset-dependent"
    explanation: str = ""
    context_checked: bool = True


class GapValidationReport(BaseModel):
    gap_id: str
    gap_title: str
    classification: Literal["explicit", "inferred"] = "explicit"
    supporting_literature: list[str] = Field(default_factory=list)
    evidence_passages: list[str] = Field(default_factory=list)
    related_papers: list[str] = Field(default_factory=list)
    existing_approaches: list[str] = Field(default_factory=list)
    reasoning: str = ""
    potential_contribution: str = ""
    technical_feasibility: str = "Medium"
    complexity: str = "Moderate"
    required_datasets: list[str] = Field(default_factory=list)
    evaluation_possibilities: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    further_searches_required: list[str] = Field(default_factory=list)


class ProblemCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"pc_{uuid4().hex[:6]}")
    title: str
    problem_statement: str
    supporting_papers: list[str] = Field(default_factory=list)
    existing_solutions: list[str] = Field(default_factory=list)
    identified_limitations: list[str] = Field(default_factory=list)
    potential_contribution: str = ""
    research_questions: list[str] = Field(default_factory=list)
    objectives: list[str] = Field(default_factory=list)
    required_resources: list[str] = Field(default_factory=list)
    suggested_methodology: str = ""
    evaluation_strategy: str = ""
    feasibility_score: float = 7.0
    contribution_score: float = 7.5


class MethodologyRecommendation(BaseModel):
    method_name: str
    suitability_reason: str
    problem_addressed: str
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evaluation_plan: str = ""
    recommended_datasets: list[str] = Field(default_factory=list)
    recommended_metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    exp_id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:6]}")
    name: str
    status: Literal["Planned", "Completed", "Simulated Example", "Actual Measured"] = "Planned"
    configuration: str = ""
    notes: str = ""


class ReproducibilityItem(BaseModel):
    criterion: str
    status: Literal["Reported", "Partial", "Missing"] = "Missing"
    evidence: str = ""
    recommendation: str = ""


class ReproducibilityCheckReport(BaseModel):
    items: list[ReproducibilityItem] = Field(default_factory=list)
    score: float = 0.0
    checked_at: str = Field(default_factory=utc_now_iso)


class PublicationSuitabilityItem(BaseModel):
    venue_name: str
    publisher_or_type: str = ""
    scope_match: str = ""
    indexing_note: str = "Verify indexing directly on publisher website"
    submission_format: str = "PDF / LaTeX / DOCX (verify on venue site)"
    word_or_page_limit: str = "Verify on official Call for Papers"
    deadlines_note: str = "Check official venue calendar"
    verification_required: bool = True
    homepage: str = ""


# ---------------------------------------------------------------------------
# Root Portable Project Model (.riq archive manifest)
# ---------------------------------------------------------------------------

class ResearchProject(BaseModel):
    project_id: str = Field(default_factory=lambda: f"proj_{uuid4().hex[:10]}")
    title: str = "Untitled Research Project"
    domain: str = "Artificial Intelligence"
    secondary_domains: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    last_exported_at: str = ""
    active_stage: str = "idea"
    completed_stages: list[str] = Field(default_factory=list)
    unsaved_changes: bool = False

    # Core research artifacts
    parsed_papers: list[ParsedPaper] = Field(default_factory=list)
    paper_analyses: dict[str, PaperAnalysis] = Field(default_factory=dict)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    multi_paper_comparison: MultiPaperComparison | None = None
    discovered_works: list[ScholarlyWork] = Field(default_factory=list)
    screening_criteria: dict[str, Any] = Field(
        default_factory=lambda: {
            "min_year": 2018,
            "required_keywords": [],
            "exclusion_keywords": [],
            "inclusion_notes": "Peer-reviewed or arXiv preprints directly addressing the research question.",
        }
    )
    cross_domain_ideas: list[CrossDomainIdea] = Field(default_factory=list)
    trend_queries: list[str] = Field(default_factory=list)
    claim_verifications: list[ClaimVerificationResult] = Field(default_factory=list)
    consistency_report: ConsistencyReport | None = None

    # Studio stage artifacts
    original_title_input: str = ""
    title_critique: str = ""
    improved_title_suggestions: list[str] = Field(default_factory=list)
    title_confirmed: bool = False
    topic_ideas: list[TopicIdea] = Field(default_factory=list)
    selected_topic: str = ""
    problem_statement: ProblemStatementData = Field(default_factory=ProblemStatementData)
    proposal: ProposalData = Field(default_factory=ProposalData)
    methodology: MethodologyDesign = Field(default_factory=MethodologyDesign)
    implementation: ImplementationPlan = Field(default_factory=ImplementationPlan)
    experiment_design: ExperimentDesign = Field(default_factory=ExperimentDesign)
    experiment_results: ExperimentResultSummary = Field(default_factory=ExperimentResultSummary)
    manuscript_sections: dict[str, ManuscriptSectionState] = Field(default_factory=dict)
    publication_prep: PublicationPrepState = Field(default_factory=PublicationPrepState)

    # Exported diagrams/SVG/Mermaid registry
    saved_diagrams: dict[str, str] = Field(default_factory=dict)
    processing_log: list[dict[str, str]] = Field(default_factory=list)

    # RESEARCHIQ 2.0 Intelligence Layer State (optional, defaults preserve .riq compatibility)
    evidence_registry: list[EvidenceItem] = Field(default_factory=list)
    contradiction_analyses: list[ContradictionAnalysisItem] = Field(default_factory=list)
    gap_validation_reports: list[GapValidationReport] = Field(default_factory=list)
    problem_candidates: list[ProblemCandidate] = Field(default_factory=list)
    methodology_recommendations: list[MethodologyRecommendation] = Field(default_factory=list)
    experiment_records: list[ExperimentRecord] = Field(default_factory=list)
    reproducibility_report: ReproducibilityCheckReport | None = None
    publication_suitability: list[PublicationSuitabilityItem] = Field(default_factory=list)
    os_assistant_history: list[dict[str, Any]] = Field(default_factory=list)
