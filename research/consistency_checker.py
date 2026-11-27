"""Feature 10: Paper Consistency Checker — objectives, methodology, experiments, results & claims."""

from __future__ import annotations

import re

from core.schemas import ConsistencyIssue, ConsistencyReport, ResearchProject


def run_consistency_check(project: ResearchProject) -> ConsistencyReport:
    """
    Analyze consistency across Problem Statement, Objectives, Methodology,
    Experiments, Results, Conclusion, and Manuscript sections.
    """
    issues: list[ConsistencyIssue] = []
    obj_map: list[dict] = []
    meth_map: list[dict] = []
    claim_links: list[dict] = []

    objectives = project.problem_statement.objectives or []
    results_text = (
        project.manuscript_sections.get("Results and Discussion").content
        if "Results and Discussion" in project.manuscript_sections
        else ""
    )
    exp_text = (
        project.manuscript_sections.get("Experimental Setup").content
        if "Experimental Setup" in project.manuscript_sections
        else ""
    )
    meth_text = (
        project.manuscript_sections.get("Proposed Methodology").content
        if "Proposed Methodology" in project.manuscript_sections
        else ""
    )
    conc_text = (
        project.manuscript_sections.get("Conclusion").content
        if "Conclusion" in project.manuscript_sections
        else ""
    )
    intro_text = (
        project.manuscript_sections.get("Introduction").content
        if "Introduction" in project.manuscript_sections
        else ""
    )

    combined_results_blob = f"{results_text} {project.experiment_results.discussion_notes}".lower()
    combined_exp_blob = f"{exp_text} {' '.join(project.experiment_design.evaluation_metrics)}".lower()

    # A. Objective-to-result mapping
    if not objectives:
        issues.append(
            ConsistencyIssue(
                category="Missing objective",
                severity="High",
                source_section="Problem Statement",
                target_section="Objectives",
                description="No formal research objectives have been defined yet.",
                recommendation="Define 2-4 measurable objectives in Research Paper Studio -> Stage 2.",
            )
        )
    else:
        for idx, obj in enumerate(objectives, start=1):
            keywords = [w for w in re.findall(r"[a-z0-9]{4,}", obj.lower()) if w not in {"using", "based", "system", "model", "research"}]
            in_exp = any(kw in combined_exp_blob for kw in keywords) if keywords else False
            in_res = any(kw in combined_results_blob for kw in keywords) if keywords else False
            has_data = bool(project.experiment_results.datasets)
            status = "Mapped" if (in_exp and (in_res or has_data)) else ("Partial" if (in_exp or in_res) else "Unmapped")
            obj_map.append(
                {
                    "Objective": f"O{idx}: {obj[:70]}",
                    "Experiment Linked": "Yes" if in_exp else "No",
                    "Result Linked": "Yes" if (in_res or has_data) else "No",
                    "Status": status,
                }
            )
            if status == "Unmapped":
                issues.append(
                    ConsistencyIssue(
                        category="Missing information",
                        severity="High",
                        source_section="Objectives",
                        target_section="Results and Discussion",
                        description=f"Objective O{idx} ('{obj[:60]}...') has no matching experiment or result evidence.",
                        recommendation=f"Add an experimental configuration and discussion paragraph explicitly evaluating Objective O{idx}.",
                    )
                )

    # B. Methodology-to-experiment mapping
    algs = project.methodology.algorithms or []
    if not algs and not project.methodology.proposed_architecture and not meth_text.strip():
        issues.append(
            ConsistencyIssue(
                category="Missing information",
                severity="High",
                source_section="Proposed Methodology",
                target_section="Experimental Setup",
                description="Proposed methodology and algorithms are not yet specified.",
                recommendation="Complete Stage 6 (Methodology & Architecture) before running experiments.",
            )
        )
    else:
        for alg in algs:
            name = alg.get("name", "Algorithm")
            mentioned_in_exp = name.lower() in combined_exp_blob or name.lower() in combined_results_blob
            meth_map.append(
                {
                    "Component": name,
                    "In Methodology": "Yes",
                    "In Experiments/Results": "Yes" if mentioned_in_exp else "Missing",
                }
            )
            if not mentioned_in_exp:
                issues.append(
                    ConsistencyIssue(
                        category="Inconsistent terminology",
                        severity="Medium",
                        source_section="Proposed Methodology",
                        target_section="Experimental Setup",
                        description=f"Algorithm/component '{name}' appears in Methodology but is not referenced in Experimental Setup or Results.",
                        recommendation=f"Ensure '{name}' is listed in baselines/configurations and evaluated in Results.",
                    )
                )

    # C. Unexplained results / unsupported claims checks
    if project.experiment_results.datasets and not results_text.strip():
        issues.append(
            ConsistencyIssue(
                category="Unexplained result",
                severity="Medium",
                source_section="Results",
                target_section="Results and Discussion",
                description="Experimental datasets are uploaded, but the Manuscript 'Results and Discussion' section is still empty.",
                recommendation="Generate or write the 'Results and Discussion' section citing your uploaded metrics.",
            )
        )

    if conc_text.strip() and not results_text.strip():
        issues.append(
            ConsistencyIssue(
                category="Unsupported claim",
                severity="High",
                source_section="Conclusion",
                target_section="Results and Discussion",
                description="Conclusion is drafted without an active 'Results and Discussion' section supporting its findings.",
                recommendation="Draft and verify 'Results and Discussion' before finalizing the Conclusion.",
            )
        )

    if intro_text.strip() and not project.discovered_works and not project.parsed_papers:
        issues.append(
            ConsistencyIssue(
                category="Unsupported claim",
                severity="Medium",
                source_section="Introduction",
                target_section="References",
                description="Introduction has text but zero uploaded PDFs or discovered scholarly references in workspace.",
                recommendation="Add supporting literature via Paper Discovery or upload reference PDFs.",
            )
        )

    # Claim links summary
    for clm in project.claim_verifications:
        claim_links.append(
            {
                "Section": clm.section_name,
                "Claim": clm.claim_text[:70],
                "Status": clm.support_status,
                "Source": clm.cited_source[:50],
            }
        )
        if clm.support_status in ("Not supported", "Unable to verify"):
            issues.append(
                ConsistencyIssue(
                    category="Unsupported claim",
                    severity="High" if clm.support_status == "Not supported" else "Medium",
                    source_section=clm.section_name or "Manuscript",
                    target_section="References",
                    description=f"Claim marked '{clm.support_status}': {clm.claim_text[:80]}",
                    recommendation="Verify cited passage or replace with an evidence-backed statement.",
                )
            )

    # Compute health score
    penalty = sum(18 if i.severity == "High" else 9 if i.severity == "Medium" else 4 for i in issues if not i.resolved)
    health_score = max(0.0, min(100.0, 100.0 - penalty))

    return ConsistencyReport(
        objective_to_result_map=obj_map,
        methodology_to_experiment_map=meth_map,
        claim_links=claim_links,
        issues=issues,
        overall_score=round(health_score, 1),
    )
