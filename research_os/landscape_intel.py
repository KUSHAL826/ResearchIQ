"""Part 2.3 & 2.4: Research Landscape Mapping & Context-Aware Contradiction Intelligence."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from core.llm_client import GeminiClient
from core.schemas import ContradictionAnalysisItem, ResearchProject
from research.trend_analyzer import ALGORITHM_KEYWORDS


def map_research_landscape(
    project: ResearchProject,
    year_min: int | None = None,
    year_max: int | None = None,
    source_filter: list[str] | None = None,
    methodology_filter: str = "",
) -> dict[str, Any]:
    """
    Build a filterable Research Landscape view over uploaded PDFs + discovered works:
    themes, algorithms, datasets, methodologies, publication activity, limitations, emerging intersections.
    """
    theme_counts: Counter[str] = Counter()
    alg_counts: Counter[str] = Counter()
    ds_counts: Counter[str] = Counter()
    meth_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    limitation_themes: list[str] = []
    intersection_pairs: Counter[str] = Counter()

    allowed_sources = set(s.lower() for s in (source_filter or []))
    meth_kw = methodology_filter.strip().lower()

    # 1. Uploaded & analysed PDFs
    if not allowed_sources or "uploaded" in allowed_sources:
        for p in project.parsed_papers:
            if year_min and p.year and p.year < year_min:
                continue
            if year_max and p.year and p.year > year_max:
                continue
            a = project.paper_analyses.get(p.paper_id)
            blob = f"{p.title} {p.hierarchical_summary}".lower()
            if meth_kw and meth_kw not in blob:
                continue
            if p.year:
                year_counts[p.year] += 1
            if a:
                for alg in a.algorithms:
                    alg_counts[alg.name] += 1
                for ds in a.datasets:
                    ds_counts[ds.name] += 1
                for step in a.methodology_steps[:3]:
                    meth_counts[step[:45]] += 1
                for lim in a.limitations[:3]:
                    limitation_themes.append(f"[{p.title[:30]}] {lim.description[:120]}")

    # 2. Discovered works
    for w in project.discovered_works:
        if w.is_duplicate:
            continue
        if allowed_sources and w.source.lower() not in allowed_sources:
            continue
        if year_min and w.year and w.year < year_min:
            continue
        if year_max and w.year and w.year > year_max:
            continue
        blob = f"{w.title} {w.abstract} {w.extracted_methodology}".lower()
        if meth_kw and meth_kw not in blob:
            continue
        if w.year:
            year_counts[w.year] += 1
        fields = [f.strip() for f in w.fields_of_study if f.strip()]
        for f in fields[:4]:
            theme_counts[f] += 1
        if len(fields) >= 2:
            pair = " × ".join(sorted(fields[:2]))
            intersection_pairs[pair] += 1
        for alg in ALGORITHM_KEYWORDS:
            if alg.lower() in blob:
                alg_counts[alg] += 1
        if w.extracted_dataset and w.extracted_dataset != "Not specified in abstract":
            for ds in [x.strip() for x in w.extracted_dataset.split(",") if x.strip()]:
                ds_counts[ds] += 1
        if w.extracted_methodology and w.extracted_methodology != "Not specified in abstract":
            meth_counts[w.extracted_methodology[:45]] += 1

    return {
        "dominant_themes": dict(theme_counts.most_common(12)),
        "frequent_algorithms": dict(alg_counts.most_common(10)),
        "popular_datasets": dict(ds_counts.most_common(10)),
        "common_methodologies": dict(meth_counts.most_common(8)),
        "publication_activity": dict(sorted(year_counts.items())),
        "emerging_intersections": dict(intersection_pairs.most_common(8)),
        "identified_limitations": limitation_themes[:10],
    }


def analyze_research_contradictions(
    project: ResearchProject,
    llm_client: GeminiClient | None = None,
) -> list[ContradictionAnalysisItem]:
    """
    Compare findings across papers while checking experimental context (datasets & metrics)
    so differing numbers across different datasets are NOT mislabelled as contradictions.
    """
    analyses = list(project.paper_analyses.values())
    results: list[ContradictionAnalysisItem] = []

    if len(analyses) >= 2 and llm_client and llm_client.is_configured:
        summaries = "\n\n".join(
            f"PAPER: {a.paper_title}\n"
            f"Datasets: {', '.join(d.name for d in a.datasets) or 'Unspecified'}\n"
            f"Algorithms: {', '.join(x.name for x in a.algorithms) or 'Unspecified'}\n"
            f"Metrics: {'; '.join(f'{r.metric}={r.raw_value or r.value} ({r.page_reference})' for r in a.results[:5])}\n"
            f"Limitations: {'; '.join(l.description for l in a.limitations[:3])}"
            for a in analyses[:6]
        )
        prompt = (
            "Compare these papers for Research Contradiction Intelligence.\n"
            "Identify conflicting results, different experimental conclusions, differing methodology assumptions, "
            "dataset-dependent findings, or contradictory claims.\n"
            "STRICT RULE: Do NOT mark differing numerical scores as contradictions unless they share the same metric/task; "
            "if datasets or assumptions differ, classify as 'Dataset-dependent' or 'Methodology assumption' and explain why.\n\n"
            f"{summaries}"
        )
        try:
            from pydantic import BaseModel, Field

            class _ContraPayload(BaseModel):
                items: list[ContradictionAnalysisItem] = Field(default_factory=list)

            payload = llm_client.generate_structured(prompt, _ContraPayload)
            if payload.items:
                project.contradiction_analyses = payload.items
                return payload.items
        except Exception:
            pass

    # Deterministic context-aware cross-paper comparison
    for i in range(len(analyses)):
        for j in range(i + 1, len(analyses)):
            a, b = analyses[i], analyses[j]
            ds_a = {d.name.lower() for d in a.datasets if d.name}
            ds_b = {d.name.lower() for d in b.datasets if d.name}
            shared_ds = ds_a & ds_b

            metrics_a = {r.metric.upper(): r for r in a.results if r.value is not None}
            metrics_b = {r.metric.upper(): r for r in b.results if r.value is not None}
            shared_metrics = set(metrics_a.keys()) & set(metrics_b.keys())

            for met in sorted(shared_metrics):
                ra, rb = metrics_a[met], metrics_b[met]
                if ra.value != rb.value:
                    if shared_ds:
                        ctype = "Conflicting results"
                        expl = (
                            f"Both studies evaluate {met} with overlapping dataset(s) ({', '.join(shared_ds)}), "
                            f"reporting {ra.raw_value} vs {rb.raw_value}. Inspect split/hyperparameter differences."
                        )
                    else:
                        ctype = "Dataset-dependent"
                        expl = (
                            f"Scores differ ({ra.raw_value} vs {rb.raw_value} on {met}), but experimental datasets differ "
                            f"({', '.join(ds_a) or 'unspecified'} vs {', '.join(ds_b) or 'unspecified'}) — not a direct contradiction."
                        )
                    results.append(
                        ContradictionAnalysisItem(
                            topic=f"{met} Comparison",
                            paper_a=a.paper_title,
                            claim_a=f"{met} = {ra.raw_value or ra.value} ({ra.page_reference})",
                            context_a=f"Datasets: {', '.join(ds_a) or 'Unspecified'}",
                            paper_b=b.paper_title,
                            claim_b=f"{met} = {rb.raw_value or rb.value} ({rb.page_reference})",
                            context_b=f"Datasets: {', '.join(ds_b) or 'Unspecified'}",
                            contradiction_type=ctype,  # type: ignore[arg-type]
                            explanation=expl,
                            context_checked=True,
                        )
                    )

    # Also compare opposite sentiment/polarity in abstracts of discovered works on shared topic
    works = [w for w in project.discovered_works if not w.is_duplicate and w.abstract][:10]
    for i in range(len(works)):
        for j in range(i + 1, len(works)):
            wa, wb = works[i], works[j]
            toks_a = set(re.findall(r"[a-z]{5,}", wa.abstract.lower()))
            toks_b = set(re.findall(r"[a-z]{5,}", wb.abstract.lower()))
            if len(toks_a & toks_b) >= 6:
                a_out = "outperform" in wa.abstract.lower()
                b_fail = any(k in wb.abstract.lower() for k in ("fail", "degrade", "underperform", "limitation"))
                if a_out and b_fail:
                    results.append(
                        ContradictionAnalysisItem(
                            topic="Empirical Generalization Tension",
                            paper_a=wa.title,
                            claim_a=wa.abstract[:180],
                            context_a=f"{wa.source} ({wa.year or 'n.d.'})",
                            paper_b=wb.title,
                            claim_b=wb.abstract[:180],
                            context_b=f"{wb.source} ({wb.year or 'n.d.'})",
                            contradiction_type="Experimental conclusion",
                            explanation="One study highlights performance gains while a topically overlapping study highlights failure or degradation conditions.",
                            context_checked=True,
                        )
                    )
                    if len(results) >= 12:
                        break
        if len(results) >= 12:
            break

    project.contradiction_analyses = results
    return results
