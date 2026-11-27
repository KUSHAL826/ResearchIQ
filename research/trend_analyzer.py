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
    Compute publication growth rates, normalized topic shares, algorithm adoption trends,
    missing-year handling, small-sample caveats, and heuristic opportunity indicators
    strictly from real session works.
    """
    unique = [w for w in works if not w.is_duplicate]
    if not unique:
        return {
            "total_works": 0,
            "dated_works": 0,
            "missing_year_count": 0,
            "small_sample_warning": "No publications available.",
            "sources": [],
            "year_range": "N/A",
            "yearly_counts": {},
            "yearly_growth_pct": {},
            "topic_counts": {},
            "topic_shares_pct": {},
            "algorithm_trends": {},
            "undated_algorithm_mentions": {},
            "topic_year_matrix": {},
            "opportunity_radar": [],
            "heuristic_disclaimer": "Opportunity scores are heuristic indicators only, not validated forecasts.",
        }

    years = [w.year for w in unique if w.year is not None and 1900 <= w.year <= 2035]
    missing_year_count = len(unique) - len(years)
    year_range = f"{min(years)} - {max(years)}" if years else "Unknown years"
    sources = sorted({w.source for w in unique})

    yearly_counts = dict(sorted(Counter(years).items()))
    yearly_growth_pct: dict[int, float | None] = {}
    prev_cnt: int | None = None
    for yr, cnt in yearly_counts.items():
        if prev_cnt is None or prev_cnt == 0:
            yearly_growth_pct[yr] = None
        else:
            yearly_growth_pct[yr] = round(((cnt - prev_cnt) / prev_cnt) * 100.0, 1)
        prev_cnt = cnt

    topic_counter: Counter[str] = Counter()
    topic_year: dict[str, Counter[int]] = defaultdict(Counter)
    alg_year: dict[str, Counter[int]] = defaultdict(Counter)
    undated_alg: Counter[str] = Counter()
    topic_citations: dict[str, list[int]] = defaultdict(list)

    for w in unique:
        blob = f"{w.title} {w.abstract}".lower()
        valid_year = w.year if (w.year is not None and 1900 <= w.year <= 2035) else None
        for f in w.fields_of_study[:4]:
            clean_f = f.strip()
            if not clean_f:
                continue
            topic_counter[clean_f] += 1
            if valid_year is not None:
                topic_year[clean_f][valid_year] += 1
            if w.citation_count is not None:
                topic_citations[clean_f].append(w.citation_count)

        for alg in ALGORITHM_KEYWORDS:
            if alg.lower() in blob:
                if valid_year is not None:
                    alg_year[alg][valid_year] += 1
                else:
                    undated_alg[alg] += 1

    total_works = len(unique)
    top_topics = [t for t, _ in topic_counter.most_common(8)]
    topic_year_matrix = {t: dict(sorted(topic_year[t].items())) for t in top_topics}
    topic_shares_pct = {
        t: round((cnt / total_works) * 100.0, 1)
        for t, cnt in topic_counter.most_common(12)
    }

    small_sample_warning = (
        f"Small-sample caution: only {total_works} unique paper(s) analysed; "
        "heuristic trend indicators may be volatile."
        if total_works < 10
        else ""
    )

    max_y = max(years) if years else None
    opportunity_radar = []
    for topic in top_topics:
        total = topic_counter[topic]
        dated_topic_total = sum(topic_year[topic].values())
        recent = sum(cnt for yr, cnt in topic_year[topic].items() if max_y and yr >= max_y - 2)
        recent_ratio = (recent / dated_topic_total) if dated_topic_total > 0 else 0.0
        avg_cites = (
            round(sum(topic_citations[topic]) / len(topic_citations[topic]), 1)
            if topic_citations[topic]
            else None
        )
        # Heuristic indicators scaled 0-10 normalized by total corpus size
        saturation_ratio = total / max(1, total_works)
        opportunity_radar.append(
            {
                "Topic": topic,
                "Publications": total,
                "Dated Publications": dated_topic_total,
                "Recent Publications": recent,
                "Corpus Share (%)": round(saturation_ratio * 100.0, 1),
                "Time Period": year_range,
                "Recent Momentum (Heuristic)": round(recent_ratio * 10.0, 2),
                "Avg Citations": avg_cites if avg_cites is not None else "N/A",
                "Gap Opportunity (Heuristic)": round(max(1.0, (1.0 - saturation_ratio) * 10.0), 2),
            }
        )

    return {
        "total_works": total_works,
        "dated_works": len(years),
        "missing_year_count": missing_year_count,
        "small_sample_warning": small_sample_warning,
        "sources": sources,
        "year_range": year_range,
        "yearly_counts": yearly_counts,
        "yearly_growth_pct": yearly_growth_pct,
        "topic_counts": dict(topic_counter.most_common(12)),
        "topic_shares_pct": topic_shares_pct,
        "algorithm_trends": {k: dict(sorted(v.items())) for k, v in alg_year.items() if sum(v.values()) > 0},
        "undated_algorithm_mentions": dict(undated_alg),
        "topic_year_matrix": topic_year_matrix,
        "opportunity_radar": opportunity_radar,
        "heuristic_disclaimer": (
            f"Heuristic indicators computed strictly from {total_works} session publication(s) "
            f"across {year_range} ({missing_year_count} missing year). Not a validated forecast."
        ),
    }
