"""Feature 4: Real scholarly paper discovery via OpenAlex, arXiv, and Semantic Scholar APIs."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Tuple

import requests

from core.config import ARXIV_BASE_URL, OPENALEX_BASE_URL, SEMANTIC_SCHOLAR_BASE_URL
from core.schemas import ScholarlyWork

REQUEST_TIMEOUT = 15


def _reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pos_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for p in positions:
                pos_word.append((int(p), word))
    if not pos_word:
        return ""
    pos_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_word)


def search_openalex(query: str, max_results: int = 10, year_from: int | None = None) -> tuple[list[ScholarlyWork], str | None]:
    """Search OpenAlex works API without fabricating any metadata."""
    params: dict[str, str | int] = {
        "search": query,
        "per-page": min(max_results, 25),
    }
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto

    try:
        resp = requests.get(f"{OPENALEX_BASE_URL}/works", params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return [], f"OpenAlex lookup failed: {exc}"

    works: list[ScholarlyWork] = []
    for item in payload.get("results", []):
        oa_id = (item.get("id") or "").rsplit("/", 1)[-1]
        title = (item.get("display_name") or item.get("title") or "").strip()
        if not title:
            continue
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in item.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        doi_raw = (item.get("doi") or "").replace("https://doi.org/", "").strip()
        primary_loc = item.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}
        venue = source_info.get("display_name") or ""
        url = primary_loc.get("landing_page_url") or item.get("doi") or item.get("id") or ""
        fields = [
            c.get("display_name", "")
            for c in (item.get("concepts") or [])[:5]
            if c.get("display_name")
        ]
        refs = [r.rsplit("/", 1)[-1] for r in (item.get("referenced_works") or [])[:15]]
        works.append(
            ScholarlyWork(
                work_id=f"oa_{oa_id}" if oa_id else f"oa_{len(works)}",
                source="openalex",
                title=title,
                authors=authors[:12],
                year=item.get("publication_year"),
                venue=venue,
                doi=doi_raw,
                url=url,
                abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
                citation_count=item.get("cited_by_count"),
                fields_of_study=fields,
                referenced_ids=[f"oa_{r}" for r in refs],
            )
        )
    return works, None


def search_arxiv(query: str, max_results: int = 10) -> tuple[list[ScholarlyWork], str | None]:
    """Search arXiv Atom API and parse real preprint metadata."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 25),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        return [], f"arXiv lookup failed: {exc}"

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    works: list[ScholarlyWork] = []
    for entry in root.findall("atom:entry", ns):
        id_text = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        arxiv_id = id_text.rsplit("/", 1)[-1]
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        if not title:
            continue
        abstract = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
        authors = [
            (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns) if c.attrib.get("term")]
        doi_el = entry.find("arxiv:doi", ns)
        doi = (doi_el.text or "").strip() if doi_el is not None else ""
        works.append(
            ScholarlyWork(
                work_id=f"arxiv_{arxiv_id}",
                source="arxiv",
                title=title,
                authors=authors[:12],
                year=year,
                venue="arXiv",
                doi=doi,
                url=id_text,
                abstract=abstract,
                citation_count=None,  # arXiv API does not provide citation counts
                fields_of_study=categories[:5],
            )
        )
    return works, None


def search_semantic_scholar(query: str, max_results: int = 10) -> tuple[list[ScholarlyWork], str | None]:
    """Search Semantic Scholar Graph API for papers, abstracts, venues, and citation counts."""
    headers = {}
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if s2_key:
        headers["x-api-key"] = s2_key

    params = {
        "query": query,
        "limit": min(max_results, 20),
        "fields": "paperId,title,authors,year,venue,externalIds,url,abstract,citationCount,fieldsOfStudy",
    }
    try:
        resp = requests.get(
            f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return [], f"Semantic Scholar lookup failed: {exc}"

    works: list[ScholarlyWork] = []
    for item in payload.get("data", []):
        title = (item.get("title") or "").strip()
        if not title:
            continue
        pid = item.get("paperId") or f"s2_{len(works)}"
        ext = item.get("externalIds") or {}
        doi = (ext.get("DOI") or "").strip()
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        works.append(
            ScholarlyWork(
                work_id=f"s2_{pid}",
                source="semantic_scholar",
                title=title,
                authors=authors[:12],
                year=item.get("year"),
                venue=item.get("venue") or "",
                doi=doi,
                url=item.get("url") or (f"https://doi.org/{doi}" if doi else ""),
                abstract=item.get("abstract") or "",
                citation_count=item.get("citationCount"),
                fields_of_study=(item.get("fieldsOfStudy") or [])[:5],
            )
        )
    return works, None


def deduplicate_works(works: list[ScholarlyWork]) -> Tuple[list[ScholarlyWork], int]:
    """
    Mark duplicate works across scholarly sources by normalized DOI or normalized title.
    Preserves all items (marking is_duplicate=True on duplicates) so PRISMA counts remain exact.
    """
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    dup_count = 0

    for w in works:
        norm_title = re.sub(r"[^a-z0-9]", "", w.title.lower())
        norm_doi = w.doi.lower().strip()
        if (norm_doi and norm_doi in seen_dois) or (norm_title and norm_title in seen_titles):
            w.is_duplicate = True
            dup_count += 1
        else:
            w.is_duplicate = False
            if norm_doi:
                seen_dois.add(norm_doi)
            if norm_title:
                seen_titles.add(norm_title)
    return works, dup_count


def discover_papers(
    query: str,
    sources: list[str] | None = None,
    per_source_limit: int = 8,
    year_from: int | None = None,
) -> tuple[list[ScholarlyWork], list[str]]:
    """Query selected scholarly APIs, merge results, and report any API errors honestly."""
    selected = sources or ["openalex", "arxiv", "semantic_scholar"]
    all_works: list[ScholarlyWork] = []
    errors: list[str] = []

    if "openalex" in selected:
        oa_works, err = search_openalex(query, max_results=per_source_limit, year_from=year_from)
        all_works.extend(oa_works)
        if err:
            errors.append(err)

    if "arxiv" in selected:
        ax_works, err = search_arxiv(query, max_results=per_source_limit)
        if year_from:
            ax_works = [w for w in ax_works if not w.year or w.year >= year_from]
        all_works.extend(ax_works)
        if err:
            errors.append(err)

    if "semantic_scholar" in selected:
        s2_works, err = search_semantic_scholar(query, max_results=per_source_limit)
        if year_from:
            s2_works = [w for w in s2_works if not w.year or w.year >= year_from]
        all_works.extend(s2_works)
        if err:
            errors.append(err)

    deduplicate_works(all_works)
    return all_works, errors
