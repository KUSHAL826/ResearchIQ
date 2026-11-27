"""Feature 8: Research Trend & Opportunity Radar grounded in real publication metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from core.schemas import ScholarlyWork

ALGORITHM_KEYWORDS = [
    "Transformer",
    "LLM",
    "RAG",
    "Graph Neural",
    "Diffusion",
    "Reinforcement Learning",
    "CNN",
    "BERT",
    "XGBoost",
    "Federated",
]


def compute_research_trends(works: list[ScholarlyWork]) -> dict[str, Any]:
    """
    Compute publication growth, topic popularity, algorithm adoption trends,
    and topic-vs-publication heat matrix strictly from real fetched/uploaded works.
    """
    unique = [w for w in works if not w.is_duplicate]
    if not unique:
        return {
            "total_works": 0,
            "sources": [],
            "year_range": "N/A",
            "yearly_counts": {},
            "topic_counts": {},
            "algorithm_trends": {},
            "topic_year_matrix": {},
            "opportunity_radar": [],
        }

    years = [w.year for w in unique if w.year]
    year_range = f"{min(years)} - {max(years)}" if years else "Unknown years"
    sources = sorted({w.source for w in unique})

    yearly_counts = dict(sorted(Counter(years).items()))
    topic_counter: Counter[str] = Counter()
    topic_year: dict[str, Counter[int]] = defaultdict(Counter)
    alg_year: dict[str, Counter[int]] = defaultdict(Counter)
    topic_citations: dict[str, list[int]] = defaultdict(list)

    for w in unique:
        blob = f"{w.title} {w.abstract}".lower()
        for f in w.fields_of_study[:4]:
            topic_counter[f] += 1
            if w.year:
                topic_year[f][w.year] += 1
            if w.citation_count is not None:
                topic_citations[f].append(w.citation_count)

        for alg in ALGORITHM_KEYWORDS:
            if alg.lower() in blob:
                if w.year:
                    alg_year[alg][w.year] += 1
                else:
                    alg_year[alg][0] += 1

    top_topics = [t for t, _ in topic_counter.most_common(8)]
    topic_year_matrix = {t: dict(sorted(topic_year[t].items())) for t in top_topics}

    # Opportunity radar: high recent growth + moderate saturation
    max_y = max(years) if years else None
    opportunity_radar = []
    for topic in top_topics:
        total = topic_counter[topic]
        recent = sum(cnt for yr, cnt in topic_year[topic].items() if max_y and yr >= max_y - 2)
        avg_cites = (
            round(sum(topic_citations[topic]) / len(topic_citations[topic]), 1)
            if topic_citations[topic]
            else 0.0
        )
        opportunity_radar.append(
            {
                "Topic": topic,
                "Publications": total,
                "Recent Momentum": round((recent / max(1, total)) * 10, 2),
                "Avg Citations": avg_cites,
                "Gap Opportunity": round(max(1.0, 10.0 - min(total, 9)), 2),
            }
        )

    return {
        "total_works": len(unique),
        "sources": sources,
        "year_range": year_range,
        "yearly_counts": yearly_counts,
        "topic_counts": dict(topic_counter.most_common(12)),
        "algorithm_trends": {k: dict(sorted(v.items())) for k, v in alg_year.items() if sum(v.values()) > 0},
        "topic_year_matrix": topic_year_matrix,
        "opportunity_radar": opportunity_radar,
    }
