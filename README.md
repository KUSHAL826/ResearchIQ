# RESEARCHIQ — Visual-First Autonomous Research Development Platform

RESEARCHIQ is an interactive, visualization-driven academic research intelligence and manuscript development platform built with **Streamlit, LangGraph, LangChain, Google Gemini, FAISS, PyMuPDF, pdfplumber, Plotly, NetworkX, PyVis, Mermaid.js, python-docx, and ReportLab**.

> **Show the research, don't just describe it.**

## Key Capabilities

- **Command Dashboard**: Live session analytics, clickable 12-stage interactive research workflow (`IDEA` → `FINAL PAPER`), and portable `.riq` project management.
- **Workspace A — Research Intelligence (10 Visual Modules)**:
  1. **Paper Intelligence**: Multi-PDF parsing, section tree, methodology & algorithm Mermaid flowcharts, dataset cards, quantitative charts, limitations, and page-level evidence viewer.
  2. **Research Gap Detective**: Author-stated (explicit) vs AI-inferred gaps, feasibility vs impact scatter, limitation matrix, and evidence-to-gap PyVis graph.
  3. **Multi-Paper Intelligence**: Side-by-side methodology, algorithm, dataset, and metric comparison with contradiction and gap overlap analysis.
  4. **Paper Discovery**: Live scholarly search via OpenAlex, arXiv, and Semantic Scholar (zero fabricated metadata).
  5. **Systematic Literature Review (PRISMA)**: Interactive Sankey PRISMA diagram computed strictly from live screening decisions, screening workbench, filterable matrix, and evidence synthesis.
  6. **Citation Network Explorer**: NetworkX centrality/PageRank analytics + interactive PyVis graph.
  7. **Cross-Domain Research Finder**: Multi-domain intersection search backed by real publication counts.
  8. **Research Trend & Opportunity Radar**: Publication growth, topic-year heatmaps, algorithm adoption lines, and opportunity radar from real metadata.
  9. **Citation Claim Verifier**: Passage-level claim verification (`Supported`, `Partially supported`, `Not supported`, `Unable to verify`). Never verifies purely on DOI existence.
  10. **Paper Consistency Checker**: Objective-to-result, methodology-to-experiment, and claim consistency audit with fix recommendations.
- **Workspace B — Three-Panel Research Paper Studio (Stages 1–12)**:
  - Guided 12-stage pipeline with approval gates, live RAG evidence drawer, Mermaid architecture editor/export, real CSV/Excel experimental analysis (zero fabricated numbers), 12-section manuscript editor, validation health dashboard, and **IEEE / Springer / ACM** exports (**DOCX, PDF, LaTeX, BibTeX**).
- **Portable `.riq` Project Archives (Zero Login / Zero Permanent DB)**:
  - Export complete workspaces (manifest, PDFs, datasets, diagrams, manuscript sections, approval states) to `.riq` ZIP archives, import with strict path-traversal validation, isolate multiple session projects, and auto-backup within session temporary storage.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # Add GEMINI_API_KEY
streamlit run app.py
```

## Privacy & Processing Notice

RESEARCHIQ does not include user accounts, logins, or a permanent database. Research state is stored in your active session and temporary files. Documents and queries sent for AI analysis or scholarly lookup are processed by Google Gemini and external scholarly APIs (OpenAlex, arXiv, Semantic Scholar).
