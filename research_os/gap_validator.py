"""Part 3: Research Gap Validation Engine — evaluates evidence, complexity, datasets, uncertainties & search needs."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from core.llm_client import GeminiClient
from core.schemas import GapValidationReport, ResearchGap, ResearchProject
from core.vector_store import SessionVectorStore


class _GapValidationPayload(BaseModel):
    reports: List[GapValidationReport] = Field(default_factory=list)


def validate_research_gaps(
    project: ResearchProject,
    vector_store: SessionVectorStore | None = None,
    llm_client: GeminiClient | None = None,
) -> list[GapValidationReport]:
    """
    Evaluate every identified research gap across 10 validation dimensions.
    Explicitly distinguishes author-stated gaps from AI-inferred opportunities
    and never claims novelty solely from sparse search results.
    """
    if not project.research_gaps:
        project.gap_validation_reports = []
        return []

    reports: list[GapValidationReport] = []
    included_titles = [
        w.title for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"
    ]
    related_titles = [
        w.title for w in project.discovered_works if not w.is_duplicate
    ][:8]

    for gap in project.research_gaps:
        passages: list[str] = []
        if gap.evidence_quote:
            passages.append(f"[{gap.supporting_publication}, {gap.page_reference}] {gap.evidence_quote}")
        if vector_store and vector_store.chunks:
            hits = vector_store.search(f"{gap.title} {gap.description}", top_k=2, llm_client=llm_client)
            for chunk, score in hits:
                passages.append(f"[{chunk.paper_title}, pp.{chunk.page_start}-{chunk.page_end}, sim={score:.2f}] {chunk.text[:280]}")

        analysis = project.paper_analyses.get(gap.paper_id)
        existing_approaches = [a.name for a in analysis.algorithms[:5]] if analysis else []
        req_datasets = [d.name for d in analysis.datasets[:4]] if analysis and analysis.datasets else ["Public benchmark dataset in target domain"]
        eval_opts = [r.metric for r in analysis.results[:4]] if analysis and analysis.results else ["Accuracy / F1 / Latency / Ablation Delta"]

        if llm_client and llm_client.is_configured:
            try:
                prompt = (
                    f"Produce a rigorous GapValidationReport for this research gap:\n"
                    f"Gap ID: {gap.gap_id}\nTitle: {gap.title}\nClassification: {gap.classification}\n"
                    f"Description: {gap.description}\nSource: {gap.supporting_publication} ({gap.page_reference})\n"
                    f"Evidence Passages:\n" + "\n".join(passages[:3]) + "\n"
                    f"Related Literature: {related_titles[:5]}\n\n"
                    f"STRICT RULES:\n"
                    f"- Distinguish established explicit gaps from AI-inferred opportunities.\n"
                    f"- List honest uncertainties and further literature searches required.\n"
                    f"- Never claim novelty is proven merely because few papers were retrieved."
                )
                rep = llm_client.generate_structured(prompt, GapValidationReport)
                rep.gap_id = gap.gap_id
                rep.gap_title = gap.title
                rep.classification = gap.classification
                if not rep.evidence_passages:
                    rep.evidence_passages = passages
                reports.append(rep)
                continue
            except Exception:
                pass

        # Deterministic grounded validation report
        is_explicit = gap.classification == "explicit"
        feasibility_lbl = "High" if gap.feasibility_score >= 7.5 else ("Medium" if gap.feasibility_score >= 5.0 else "Low")
        uncertainties = [
            "Published work outside the uploaded/fetched session corpus may already address part of this gap.",
            "True benchmark difficulty depends on dataset availability and baseline reproducibility.",
        ]
        if not is_explicit:
            uncertainties.insert(0, "AI-inferred opportunity — not explicitly stated by source authors; requires targeted literature confirmation.")

        reports.append(
            GapValidationReport(
                gap_id=gap.gap_id,
                gap_title=gap.title,
                classification=gap.classification,
                supporting_literature=[f"{gap.supporting_publication} ({gap.page_reference})"] + included_titles[:3],
                evidence_passages=passages or ["No verbatim quote extracted; inspect source paper directly."],
                related_papers=related_titles[:5],
                existing_approaches=existing_approaches or ["See methodology breakdown in Paper Intelligence"],
                reasoning=(
                    f"{'Author-stated limitation verified in source text' if is_explicit else 'Inferred structural/evaluation gap across analysed papers'}: "
                    f"{gap.technical_significance or gap.description}"
                ),
                potential_contribution=gap.potential_improvement or f"Address '{gap.title}' with controlled empirical evaluation.",
                technical_feasibility=feasibility_lbl,
                complexity=gap.implementation_difficulty,
                required_datasets=req_datasets,
                evaluation_possibilities=eval_opts,
                uncertainties=uncertainties,
                further_searches_required=[
                    f"Search OpenAlex/Semantic Scholar for recent benchmarks addressing: {gap.title[:60]}",
                    f"Verify recent surveys in {project.domain} (last 3 years) before claiming novelty.",
                ],
            )
        )

    project.gap_validation_reports = reports
    return reports
