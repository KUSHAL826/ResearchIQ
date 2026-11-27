"""Single LangGraph Master Orchestrator coordinating all RESEARCHIQ capabilities via @tool and Pydantic schemas."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from core.llm_client import GeminiClient
from core.schemas import ResearchProject
from core.vector_store import SessionVectorStore
from research.citation_network import analyze_graph_metrics, build_citation_graph
from research.citation_verifier import verify_claim_against_evidence
from research.consistency_checker import run_consistency_check
from research.paper_discovery import discover_papers
from research.paper_intelligence import analyze_paper_intelligence
from research.research_gap import detect_research_gaps
from research.systematic_review import compute_prisma_counts


class OrchestratorState(TypedDict, total=False):
    task_type: str
    query: str
    params: dict[str, Any]
    approved_context: str
    retrieved_evidence: list[dict[str, Any]]
    result: Any
    trace: list[str]
    error: str


def build_approved_context_summary(project: ResearchProject) -> str:
    """
    Assemble single-session research context using ONLY approved upstream outputs
    (or active project metadata) so unapproved drafts never silently leak downstream.
    """
    lines = [
        f"Project Title: {project.title}",
        f"Primary Domain: {project.domain}",
        f"Selected Topic: {project.selected_topic or 'Not yet selected'}",
    ]
    if project.problem_statement.approved:
        lines.append(f"[APPROVED Problem Statement]: {project.problem_statement.problem_definition}")
        if project.problem_statement.objectives:
            lines.append("[APPROVED Objectives]: " + "; ".join(project.problem_statement.objectives))
    if project.proposal.approved:
        lines.append(f"[APPROVED Proposal Summary]: {project.proposal.methodology_summary[:400]}")
    if project.methodology.approved:
        lines.append(f"[APPROVED Methodology]: {project.methodology.proposed_architecture[:450]}")
    if project.implementation.approved:
        lines.append(f"[APPROVED Implementation]: {len(project.implementation.modules)} modules planned")
    if project.experiment_design.approved:
        lines.append("[APPROVED Metrics]: " + ", ".join(project.experiment_design.evaluation_metrics))
    if project.experiment_results.approved and project.experiment_results.datasets:
        ds = project.experiment_results.datasets[0]
        lines.append(f"[APPROVED Results Dataset]: {ds.filename} ({ds.row_count} rows)")

    included = [w for w in project.discovered_works if not w.is_duplicate and w.screening_status == "included"]
    if included:
        lines.append(
            "[Included Literature]: "
            + "; ".join(f"{w.title} ({w.year or 'N/A'})" for w in included[:8])
        )
    if project.research_gaps:
        lines.append(
            "[Identified Gaps]: "
            + "; ".join(f"{g.title} [{g.classification}, {g.page_reference}]" for g in project.research_gaps[:6])
        )
    return "\n".join(lines)


def create_session_tools(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
) -> list:
    """Create session-bound LangChain @tool functions for the Master Orchestrator."""

    @tool
    def search_session_rag_tool(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant evidence passages with exact paper titles and page citations from uploaded PDFs."""
        hits = vector_store.search(query, top_k=top_k, llm_client=llm_client)
        return [
            {
                "paper_title": c.paper_title,
                "section": c.section,
                "pages": f"pp. {c.page_start}-{c.page_end}",
                "score": round(score, 3),
                "excerpt": c.text[:500],
            }
            for c, score in hits
        ]

    @tool
    def discover_scholarly_works_tool(query: str, per_source: int = 5) -> dict[str, Any]:
        """Search OpenAlex, arXiv, and Semantic Scholar for scholarly publications."""
        works, errs = discover_papers(query, per_source_limit=per_source)
        return {
            "count": len(works),
            "errors": errs,
            "titles": [f"{w.title} ({w.year or 'N/A'}, {w.source})" for w in works[:10]],
        }

    @tool
    def prisma_status_tool() -> dict[str, Any]:
        """Compute live PRISMA systematic review counts from current screening data."""
        counts = compute_prisma_counts(project.discovered_works, project.parsed_papers)
        return counts.model_dump()

    @tool
    def citation_network_metrics_tool() -> dict[str, Any]:
        """Analyze the session citation network using NetworkX centrality and PageRank."""
        G = build_citation_graph(project.discovered_works, project.parsed_papers, project.paper_analyses)
        return analyze_graph_metrics(G)

    @tool
    def detect_gaps_tool() -> list[dict[str, Any]]:
        """Detect explicit and inferred research gaps across uploaded PDFs."""
        gaps = detect_research_gaps(project.parsed_papers, project.paper_analyses, llm_client=llm_client)
        return [g.model_dump() for g in gaps]

    @tool
    def verify_claim_tool(claim_text: str, cited_source: str = "") -> dict[str, Any]:
        """Verify a manuscript claim against uploaded PDF passages and discovered abstracts."""
        res = verify_claim_against_evidence(
            claim_text=claim_text,
            cited_source=cited_source,
            section_name="Query",
            vector_store=vector_store,
            discovered_works=project.discovered_works,
            llm_client=llm_client,
        )
        return res.model_dump()

    @tool
    def consistency_check_tool() -> dict[str, Any]:
        """Check consistency across objectives, methodology, experiments, results, and manuscript sections."""
        rep = run_consistency_check(project)
        return rep.model_dump()

    return [
        search_session_rag_tool,
        discover_scholarly_works_tool,
        prisma_status_tool,
        citation_network_metrics_tool,
        detect_gaps_tool,
        verify_claim_tool,
        consistency_check_tool,
    ]


class MasterResearchOrchestrator:
    """
    Single LangGraph StateGraph orchestrator coordinating document intelligence,
    discovery, screening, gaps, studio generation, verification, and consistency checks.
    """

    def __init__(
        self,
        project: ResearchProject,
        vector_store: SessionVectorStore,
        llm_client: GeminiClient,
    ):
        self.project = project
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.tools = create_session_tools(project, vector_store, llm_client)
        self.tool_map = {t.name: t for t in self.tools}
        self.graph = self._compile_graph()

    def _compile_graph(self):
        builder = StateGraph(OrchestratorState)

        def gather_context(state: OrchestratorState) -> OrchestratorState:
            trace = list(state.get("trace") or [])
            trace.append("Context assembled from approved stages")
            approved_ctx = build_approved_context_summary(self.project)
            return {**state, "approved_context": approved_ctx, "trace": trace}

        def retrieve_evidence(state: OrchestratorState) -> OrchestratorState:
            trace = list(state.get("trace") or [])
            query = state.get("query") or self.project.selected_topic or self.project.title
            evidence: list[dict[str, Any]] = []
            if self.vector_store and self.vector_store.chunks:
                hits = self.vector_store.search(query, top_k=5, llm_client=self.llm_client)
                evidence = [
                    {
                        "paper_title": c.paper_title,
                        "section": c.section,
                        "pages": f"pp. {c.page_start}-{c.page_end}",
                        "score": round(score, 3),
                        "text": c.text,
                    }
                    for c, score in hits
                ]
                trace.append(f"Retrieved {len(evidence)} RAG passages from FAISS")
            else:
                trace.append("No PDF chunks in FAISS; using scholarly metadata & studio context")
            return {**state, "retrieved_evidence": evidence, "trace": trace}

        def execute_task(state: OrchestratorState) -> OrchestratorState:
            trace = list(state.get("trace") or [])
            task = state.get("task_type", "")
            params = state.get("params") or {}
            result: Any = None

            try:
                if task == "paper_intelligence":
                    paper = params["paper"]
                    result = analyze_paper_intelligence(paper, llm_client=self.llm_client)
                    trace.append(f"Paper intelligence completed for '{paper.title[:40]}'")
                elif task == "gap_detection":
                    result = detect_research_gaps(
                        self.project.parsed_papers,
                        self.project.paper_analyses,
                        llm_client=self.llm_client,
                    )
                    trace.append(f"Identified {len(result)} research gaps")
                elif task == "discovery":
                    works, errs = discover_papers(
                        query=state.get("query", self.project.domain),
                        sources=params.get("sources"),
                        per_source_limit=params.get("limit", 8),
                        year_from=params.get("year_from"),
                    )
                    result = {"works": works, "errors": errs}
                    trace.append(f"Discovered {len(works)} scholarly records")
                elif task == "consistency":
                    result = run_consistency_check(self.project)
                    trace.append(f"Consistency check completed (score={result.overall_score})")
                elif task == "custom_fn" and callable(params.get("fn")):
                    fn: Callable[..., Any] = params["fn"]
                    result = fn(
                        project=self.project,
                        approved_context=state.get("approved_context", ""),
                        retrieved_evidence=state.get("retrieved_evidence", []),
                        llm_client=self.llm_client,
                    )
                    trace.append(f"Executed studio task '{params.get('label', 'custom')}'")
                else:
                    result = {"status": "no_op", "task_type": task}
                    trace.append(f"Unhandled task_type '{task}'")
                return {**state, "result": result, "trace": trace}
            except Exception as exc:
                trace.append(f"Task failed: {exc}")
                return {**state, "error": str(exc), "trace": trace}

        builder.add_node("gather_context", gather_context)
        builder.add_node("retrieve_evidence", retrieve_evidence)
        builder.add_node("execute_task", execute_task)

        builder.add_edge(START, "gather_context")
        builder.add_edge("gather_context", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "execute_task")
        builder.add_edge("execute_task", END)
        return builder.compile()

    def run(
        self,
        task_type: str,
        query: str = "",
        params: dict[str, Any] | None = None,
    ) -> OrchestratorState:
        """Invoke the compiled LangGraph workflow synchronously."""
        initial: OrchestratorState = {
            "task_type": task_type,
            "query": query,
            "params": params or {},
            "trace": [],
        }
        return self.graph.invoke(initial)
