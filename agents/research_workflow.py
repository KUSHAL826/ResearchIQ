"""Workflow coordination helpers connecting UI events to the LangGraph Master Orchestrator."""

from __future__ import annotations

from typing import Any, Callable

from agents.master_orchestrator import MasterResearchOrchestrator
from core.llm_client import GeminiClient
from core.schemas import ResearchProject, utc_now_iso
from core.vector_store import SessionVectorStore


def run_orchestrated_task(
    project: ResearchProject,
    vector_store: SessionVectorStore,
    llm_client: GeminiClient,
    task_type: str,
    query: str = "",
    params: dict[str, Any] | None = None,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a task through the single MasterResearchOrchestrator and log trace events."""
    if stage_callback:
        stage_callback("Research retrieved")

    orchestrator = MasterResearchOrchestrator(
        project=project,
        vector_store=vector_store,
        llm_client=llm_client,
    )
    if stage_callback:
        stage_callback("Analysis performed")

    state = orchestrator.run(task_type=task_type, query=query, params=params)
    for step in state.get("trace", []):
        project.processing_log.append({"timestamp": utc_now_iso(), "event": step})
    project.processing_log = project.processing_log[-60:]

    if stage_callback:
        stage_callback("Results generated")

    return dict(state)
