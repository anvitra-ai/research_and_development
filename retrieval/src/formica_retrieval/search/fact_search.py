"""Searching the graph by what facts SAY, not by which relation they carry.

The template Cypher can only return edges whose r.name appears in prop1_list,
which is built from a hand-maintained keyword map -- so a relation nobody wrote
a keyword for is invisible to it. These searches have no such blind spot: they
rank r.fact text against the query by embedding similarity (or, as a last
resort, keyword overlap).

Used two ways by the pipeline: as a supplement merged into every broad template
result, and as the sole retrieval path for cross-company questions that resolve
no entity at all.
"""

from __future__ import annotations

import re
from typing import Any

from sentence_transformers import util

from .cypher_templates import _TEXT_SEARCH_CYPHER, _TEXT_SEARCH_STOPWORDS
from .relation_names import _with_casing_variants


def _run_cypher(driver, cypher: str, params: dict[str, Any]) -> list:
    with driver.session() as s:
        return list(s.run(cypher, **params))


def semantic_fact_search(
    query: str,
    fact_index: tuple[list[dict[str, Any]], Any],
    embedder,
    top_k: int = 25,
    min_similarity: float = 0.25,
    subject_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank the graph's facts against the query by embedding similarity.

    Replaces the keyword-overlap scan below as the primary last-resort search.
    Substring scoring can't match "merger" to "amalgamated"/"merged", has no
    notion of which words carry the question's meaning, and in practice kept
    degrading to 1-keyword matches that pulled in facts sharing only an
    incidental word ("its history of governance disputes" matching on
    "history"). Cosine similarity over the same sentence-transformer already
    used for entity linking ranks on meaning instead, so near-misses in wording
    still surface and incidental word overlap doesn't.

    `subject_names` restricts results to those companies. When the query DID
    resolve an entity, searching the whole graph would happily return a better-
    worded fact about the wrong bank, so scoping to the companies the question
    is actually about is what makes this usable as a supplement to the template
    query rather than only as a last resort. Because the filter is applied after
    ranking, top_k is widened proportionally so a narrow scope still gets a full
    slate of candidates rather than whatever few survive the cut.
    """
    rows, fact_emb = fact_index
    if not rows or fact_emb is None:
        return []
    search_k = min(len(rows), top_k * 8 if subject_names else top_k)
    query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(query_emb, fact_emb, top_k=search_k)[0]
    wanted = {n.lower() for n in subject_names} if subject_names else None
    out: list[dict[str, Any]] = []
    for hit in hits:
        score = float(hit["score"])
        if score < min_similarity:
            continue
        row = dict(rows[hit["corpus_id"]])
        if wanted is not None and str(row.get("subject_name") or "").lower() not in wanted:
            continue
        row["score"] = round(score, 3)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


def cross_company_search(
    query: str,
    fact_index: tuple[list[dict[str, Any]], Any],
    embedder,
    top_k: int = 200,
    max_per_company: int = 5,
    min_similarity: float = 0.28,
) -> list[dict[str, Any]]:
    """Facts across MANY companies, diversified -- not the top-k most similar.

    semantic_fact_search ranks purely by cosine similarity, which is the right
    operation for "find the fact that answers this" but the wrong one for
    "find every company for which this holds": a 39-bank enumeration question
    ("which banks have LIC as a shareholder?", "which banks are under
    regulatory scrutiny?") plausibly has its true answer set scattered across
    the whole similarity ranking, and truncating to the top few dozen
    concentrates on however many companies happen to phrase the fact closest
    to the query -- silently dropping banks whose matching fact is worded
    differently but is no less true.

    This caps how many facts any single company can contribute
    (`max_per_company`) before the overall cap, so the candidate set actually
    samples across companies instead of one or two dominating every slot. It is
    deliberately looser than semantic_fact_search's floor and cap: recall over
    precision, on the assumption that a synthesis step downstream (an LLM
    reading the whole candidate set) is far better positioned to say "no, that
    one doesn't actually qualify" than a similarity cutoff is to guess it.
    """
    rows, fact_emb = fact_index
    if not rows or fact_emb is None:
        return []
    query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    # Ask for far more hits than top_k so the per-company cap has enough
    # candidates to diversify from, rather than exhausting the ranked list on
    # one company's cluster of near-duplicate facts.
    search_k = min(len(rows), max(top_k * 4, 400))
    hits = util.semantic_search(query_emb, fact_emb, top_k=search_k)[0]

    per_company: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for hit in hits:
        score = float(hit["score"])
        if score < min_similarity:
            continue
        row = dict(rows[hit["corpus_id"]])
        company = str(row.get("subject_name") or "")
        if per_company.get(company, 0) >= max_per_company:
            continue
        per_company[company] = per_company.get(company, 0) + 1
        row["score"] = round(score, 3)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


_COMPANY_PROFILE_CYPHER = (
    "MATCH (c:Company) WHERE c.name IN $names AND c.summary IS NOT NULL\n"
    "RETURN c.name AS subject_name, c.summary AS profile"
)

_ENTITY_PROFILE_CYPHER = (
    "MATCH (n:Entity {uuid: $uuid})\n"
    "WHERE n.summary IS NOT NULL AND n.summary <> ''\n"
    "RETURN n.name AS subject_name, n.summary AS profile"
)


def entity_profile_rows(driver, uuid: str | None) -> list[dict[str, Any]]:
    """One node's `summary`, for the templates that return a NODE rather than edges.

    attribute_lookup answers by listing a node's scalar properties, and
    _ATTRIBUTE_LOOKUP_EXCLUDED_KEYS deliberately drops `summary` from that list
    (it is prose, not a scalar attribute). The effect was that the single
    richest field on the node was unreachable by the one strategy whose whole
    job is to describe that node: "Who is Uday Kotak?" answered "Uday Kotak is a
    Promoter" from role_category, while the node's own summary said "a principal
    member of the Kotak family promoter group".

    Not Company-scoped, unlike company_profile_rows -- summaries exist on most
    entity types (Driver 138, Company 86, Person 79, Geography 73, Metric 71,
    ...), and it is precisely the non-Company ones that attribute_lookup
    describes most poorly, since they have few scalar properties to fall back on.
    """
    if not uuid:
        return []
    return [dict(r) for r in _run_cypher(driver, _ENTITY_PROFILE_CYPHER, {"uuid": uuid})]


def company_profile_rows(driver, company_names: set[str]) -> list[dict[str, Any]]:
    """Each named company's node `summary`, as rows the summariser can read.

    Graphiti writes a dense per-node profile that the retrieval layer has never
    used -- attribute_lookup explicitly excludes `summary` (see
    _ATTRIBUTE_LOOKUP_EXCLUDED_KEYS), so it was unreachable by any query path.
    It routinely carries figures that were never extracted into their own edges,
    which is exactly the gap that leaves an otherwise-answerable question with
    "the retrieved facts do not cover it".
    """
    if not company_names:
        return []
    return [dict(r) for r in _run_cypher(driver, _COMPANY_PROFILE_CYPHER, {"names": sorted(company_names)})]


def per_company_top_facts(
    query: str,
    fact_index: tuple[list[dict[str, Any]], Any],
    embedder,
    per_company: int = 2,
    max_companies: int = 60,
) -> list[dict[str, Any]]:
    """Each company's own best-matching facts, with NO global ranking or floor.

    cross_company_search diversifies but still ranks globally and drops anything
    under a similarity floor -- which is fatal for enumeration questions
    ("which banks are under regulatory scrutiny?", "which banks have LIC as a
    shareholder?"). A bank whose qualifying fact is phrased unusually scores
    below the floor, gets dropped before the summariser ever sees it, and is
    therefore *structurally* unable to appear in the answer no matter how good
    the model is. The answer set is silently truncated to whichever banks happen
    to phrase things closest to the question.

    This instead asks, per company independently: "what are YOUR most relevant
    facts for this question?" -- so every company is represented and the
    filtering decision moves to the summariser, which can actually read the
    fact and judge whether it qualifies. Recall over precision, deliberately:
    for an enumeration question a missing bank is a wrong answer, whereas an
    extra irrelevant one is something the model can discard.
    """
    rows, fact_emb = fact_index
    if not rows or fact_emb is None:
        return []

    query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(query_emb, fact_emb)[0]

    by_company: dict[str, list[tuple[float, int]]] = {}
    for idx, row in enumerate(rows):
        company = str(row.get("subject_name") or "")
        if not company:
            continue
        by_company.setdefault(company, []).append((float(sims[idx]), idx))

    out: list[dict[str, Any]] = []
    for company in sorted(by_company)[:max_companies]:
        best = sorted(by_company[company], key=lambda pair: pair[0], reverse=True)[:per_company]
        for score, idx in best:
            row = dict(rows[idx])
            row["score"] = round(score, 3)
            out.append(row)
    return out


def semantic_relation_names(
    query: str,
    fact_index: tuple[list[dict[str, Any]], Any],
    embedder,
    subject_names: set[str] | None = None,
    max_relations: int = 5,
    min_similarity: float = 0.30,
) -> list[str]:
    """Relation names the query semantically touches, read off the fact index.

    RELATION_KEYWORDS is a hand-maintained map of 56 relations to 458 keywords,
    and every re-ingestion invalidates part of it: the graph currently carries 72
    distinct r.name values, 24 of which (206 edges) the map has never heard of.
    A relation missing from the map is completely invisible to the Cypher
    templates, because prop1_list is built purely from keyword hits.

    Rather than maintain a second hand-written index of relation names, this
    reads the relations off the facts that the embedding search already ranked
    highest: if the most on-topic facts for this query are HAS_LICENSE edges,
    then HAS_LICENSE belongs in prop1_list whether or not anyone wrote a keyword
    for it. Keyword matches are kept and these are unioned in, so this can only
    widen the filter, never drop a relation the map got right.
    """
    hits = semantic_fact_search(
        query,
        fact_index,
        embedder,
        top_k=40,
        min_similarity=min_similarity,
        subject_names=subject_names,
    )
    ordered: list[str] = []
    for hit in hits:
        name = hit.get("relationship")
        if name and name not in ordered:
            ordered.append(str(name))
        if len(ordered) >= max_relations:
            break
    return _with_casing_variants(ordered)


def text_search_fallback(driver, query: str) -> list:
    """Last-resort keyword scan over every company's fact text.

    For cross-company questions that name no company and no gazetteer-resolvable
    topic at all ("which banks have undergone a merger or amalgamation", "which
    banks disclose a Chairman separate from the MD & CEO role"), entity
    resolution returns nothing and the whole pipeline used to bail out with
    error="no_entities_resolved" and an empty summary -- a guaranteed failure on
    ~3% of this benchmark. Scoring facts by how many of the query's content
    words they contain at least gives the summarizer real, on-topic facts to
    work from. Deliberately requires 2+ keyword hits so this stays a targeted
    scan rather than dumping the graph.
    """
    words = {w for w in re.findall(r"[a-z][a-z\-]{3,}", query.lower()) if w not in _TEXT_SEARCH_STOPWORDS}
    if len(words) < 2:
        return []
    keywords = sorted(words)
    rows = _run_cypher(driver, _TEXT_SEARCH_CYPHER, {"keywords": keywords, "min_score": 2})
    if not rows:
        rows = _run_cypher(driver, _TEXT_SEARCH_CYPHER, {"keywords": keywords, "min_score": 1})
    return rows
