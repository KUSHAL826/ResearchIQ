"""Feature 1: Paper Intelligence — visual section breakdown, algorithms, datasets, results, limitations."""

from __future__ import annotations

import re

from core.llm_client import GeminiClient
from core.schemas import (
    AlgorithmItem,
    DatasetItem,
    LimitationItem,
    MetricResultItem,
    PaperAnalysis,
    ParsedPaper,
)


def _sanitize_mermaid_label(text: str) -> str:
    cleaned = re.sub(r'["`\[\]\(\){}]', " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:48] or "Step"


def build_methodology_mermaid(steps: list[str], title: str = "Methodology") -> str:
    """Create a valid Mermaid flowchart from methodology steps."""
    if not steps:
        steps = ["Problem Input", "Preprocessing", "Core Model", "Evaluation"]
    lines = ["flowchart TD"]
    for idx, step in enumerate(steps[:10]):
        node_id = f"M{idx}"
        label = _sanitize_mermaid_label(step)
        lines.append(f'    {node_id}["{idx + 1}. {label}"]')
        if idx > 0:
            lines.append(f"    M{idx - 1} --> {node_id}")
    return "\n".join(lines)


def build_algorithm_mermaid(algorithms: list[AlgorithmItem]) -> str:
    """Create a Mermaid relationship diagram connecting algorithms, inputs, and outputs."""
    if not algorithms:
        return 'flowchart LR\n    IN["Input Features"] --> ALG["Primary Model"] --> OUT["Predictions / Metrics"]'
    lines = ["flowchart LR"]
    for idx, alg in enumerate(algorithms[:6]):
        aid = f"A{idx}"
        aname = _sanitize_mermaid_label(alg.name)
        inp = _sanitize_mermaid_label(alg.inputs or "Paper Data")
        out = _sanitize_mermaid_label(alg.outputs or "Representation / Output")
        lines.append(f'    I{idx}["{inp}"] --> {aid}["{aname}"] --> O{idx}["{out}"]')
        if idx > 0:
            lines.append(f"    O{idx - 1} -.-> I{idx}")
    return "\n".join(lines)


def analyze_paper_intelligence(
    paper: ParsedPaper,
    llm_client: GeminiClient | None = None,
) -> PaperAnalysis:
    """
    Perform structured intelligence extraction on a ParsedPaper using RAG chunks & Gemini,
    or transparent heuristic extraction from actual parsed sections when API key is not set.
    """
    section_snippets = "\n\n".join(
        f"[{sec.canonical_name} | pp.{sec.start_page}-{sec.end_page}]\n{sec.content[:2200]}"
        for sec in paper.sections[:8]
    )

    if llm_client and llm_client.is_configured:
        try:
            prompt = (
                f"Analyze the following research paper and produce a structured PaperAnalysis.\n"
                f"Paper ID: {paper.paper_id}\nTitle: {paper.title}\n\n"
                f"REQUIREMENTS:\n"
                f"- Only report algorithms, datasets, metrics, and limitations explicitly supported by the text below.\n"
                f"- Include exact page_reference strings like 'p. 4' or 'pp. 3-5'.\n"
                f"- For results, include numeric value when stated.\n"
                f"- Provide 4-7 concise methodology_steps.\n\n"
                f"EXTRACTED PAPER CONTENT:\n{section_snippets[:16000]}"
            )
            analysis = llm_client.generate_structured(prompt, PaperAnalysis)
            analysis.paper_id = paper.paper_id
            analysis.paper_title = paper.title
            if not analysis.mermaid_methodology:
                analysis.mermaid_methodology = build_methodology_mermaid(analysis.methodology_steps)
            if not analysis.mermaid_algorithms:
                analysis.mermaid_algorithms = build_algorithm_mermaid(analysis.algorithms)
            return analysis
        except Exception:
            pass

    # Transparent rule-based extraction from actual parsed pages & sections
    sec_map = {s.canonical_name: s for s in paper.sections}
    abstract_sec = sec_map.get("Abstract") or sec_map.get("Introduction")
    meth_sec = sec_map.get("Methodology")
    exp_sec = sec_map.get("Experiments")
    res_sec = sec_map.get("Results")
    conc_sec = sec_map.get("Conclusion")

    problem = abstract_sec.summary if abstract_sec else paper.hierarchical_summary[:320]

    # Methodology steps from sentences in Methodology section
    meth_steps: list[str] = []
    if meth_sec:
        sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", meth_sec.content))
        meth_steps = [s[:95] for s in sents if len(s) > 25][:6]
    if not meth_steps:
        meth_steps = [f"{s.canonical_name} (pp.{s.start_page}-{s.end_page})" for s in paper.sections[:5]]

    # Heuristic algorithm detection
    full_text = "\n".join(p.text for p in paper.pages)
    alg_candidates = [
        "Transformer", "BERT", "RoBERTa", "GPT", "ResNet", "XGBoost", "Random Forest",
        "SVM", "LSTM", "GRU", "Graph Neural Network", "GNN", "FAISS", "BM25", "PPO", "DQN",
    ]
    algorithms: list[AlgorithmItem] = []
    for cand in alg_candidates:
        for p in paper.pages:
            if re.search(rf"\b{re.escape(cand)}\b", p.text, re.IGNORECASE):
                algorithms.append(
                    AlgorithmItem(
                        name=cand,
                        role="Referenced/utilized model or method in document",
                        inputs="Document/task inputs",
                        outputs="Task representation or predictions",
                        page_reference=f"p. {p.page_number}",
                    )
                )
                break

    # Heuristic dataset detection
    datasets: list[DatasetItem] = []
    ds_matches = re.finditer(r"\b([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z0-9][A-Za-z0-9\-]{1,})?)\s+(?:dataset|benchmark|corpus)\b", full_text)
    seen_ds: set[str] = set()
    for m in ds_matches:
        dname = m.group(1).strip()
        if dname.lower() not in seen_ds and len(dname) > 2:
            seen_ds.add(dname.lower())
            page_num = next((p.page_number for p in paper.pages if dname in p.text), 1)
            datasets.append(DatasetItem(name=dname, page_reference=f"p. {page_num}"))
        if len(datasets) >= 8:
            break

    # Metric extraction from Results/Experiments
    results: list[MetricResultItem] = []
    metric_re = re.compile(
        r"\b(accuracy|f1[\-\s]?score|precision|recall|auc|roc\-auc|rmse|mae|bleu|rouge)\b[^.\n]{0,35}?(\d{1,3}(?:\.\d+)?\s*%?)",
        re.IGNORECASE,
    )
    target_pages = [
        p for p in paper.pages
        if not res_sec or (res_sec.start_page <= p.page_number <= res_sec.end_page) or (exp_sec and exp_sec.start_page <= p.page_number <= exp_sec.end_page)
    ] or paper.pages
    for p in target_pages:
        for m in metric_re.finditer(p.text):
            raw_v = m.group(2).strip()
            try:
                num_v = float(raw_v.replace("%", ""))
            except ValueError:
                num_v = None
            results.append(
                MetricResultItem(
                    metric=m.group(1).upper(),
                    value=num_v,
                    raw_value=raw_v,
                    model_or_method="Reported in text",
                    page_reference=f"p. {p.page_number}",
                )
            )
            if len(results) >= 12:
                break
        if len(results) >= 12:
            break

    # Limitations detection
    limitations: list[LimitationItem] = []
    lim_re = re.compile(r"([^.\n]*\b(?:limitation|drawback|bottleneck|challenge|future work|shortcoming|fail|struggle|constraint)\b[^.\n]*\.)", re.IGNORECASE)
    for p in paper.pages:
        for m in lim_re.finditer(p.text):
            sent = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(sent) > 25:
                limitations.append(
                    LimitationItem(
                        description=sent,
                        severity="Medium",
                        page_reference=f"p. {p.page_number}",
                        source_quote=sent[:240],
                    )
                )
            if len(limitations) >= 8:
                break
        if len(limitations) >= 8:
            break

    return PaperAnalysis(
        paper_id=paper.paper_id,
        paper_title=paper.title,
        problem_statement=problem,
        key_contributions=[s.summary for s in paper.sections[:3] if s.summary],
        methodology_steps=meth_steps,
        algorithms=algorithms[:8],
        datasets=datasets,
        results=results,
        limitations=limitations,
        mermaid_methodology=build_methodology_mermaid(meth_steps),
        mermaid_algorithms=build_algorithm_mermaid(algorithms),
    )
