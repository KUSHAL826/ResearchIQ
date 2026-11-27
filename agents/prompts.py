"""Structured academic prompts enforcing evidence grounding and zero fabrication."""

SYSTEM_RESEARCH_ARCHITECT = (
    "You are RESEARCHIQ, an academic research intelligence and manuscript engineering system. "
    "Follow strict scientific integrity: "
    "1) Never fabricate citations, DOIs, citation counts, datasets, or experimental numbers. "
    "2) Ground every empirical claim in the provided session evidence (uploaded PDFs with page numbers, "
    "retrieved scholarly metadata, or uploaded CSV/Excel measurements). "
    "3) Clearly distinguish established evidence from proposed ideas or inferred hypotheses. "
    "4) Write concise, rigorous academic prose suitable for IEEE, Springer, or ACM manuscripts."
)

SECTION_WRITING_GUIDANCE = {
    "Abstract": "Summarize the problem, research gap, proposed methodology, experimental setup, key verified findings, and main contribution in 180-250 words.",
    "Introduction": "Establish domain context, motivate the problem with grounded citations, state the research gap, and list clear objectives and contributions.",
    "Literature Review": "Synthesize included scholarly works and uploaded papers thematically, comparing methodologies, datasets, and limitations.",
    "Research Gap": "Present explicit author-stated limitations and inferred gaps with supporting publications and page/source references.",
    "Objectives": "State crisp, measurable research objectives aligned with the problem statement and hypotheses.",
    "Proposed Methodology": "Detail the system architecture, mathematical formulation, algorithms, data flow, and design rationale.",
    "Implementation": "Describe the technology stack, preprocessing pipeline, module organization, and reproducibility details.",
    "Experimental Setup": "Specify baselines, evaluation metrics, datasets, hyperparameter configurations, and ablation study design.",
    "Results and Discussion": "Report ONLY verified metrics from uploaded experimental datasets. Analyze improvements, trade-offs, and error patterns.",
    "Conclusion": "Summarize how each objective was addressed, highlight core findings, and acknowledge study limitations.",
    "Future Scope": "Propose concrete, evidence-grounded extensions addressing remaining limitations.",
    "References": "Compile formatted academic references strictly from verified session papers and discovered works.",
}
