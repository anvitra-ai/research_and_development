"""Score how well retrieved KG rows actually answer the query.

The execution cascade in template_resolver historically accepted the first
strategy that returned a non-empty result, with no check that the rows had
anything to do with the question. Measured on the v13 benchmark, that is the
single largest failure mode: of 361 non-Relevant queries, 111 returned rows
containing NONE of the ground-truth answer and another 147 only part of it --
against just 33 where the full answer was retrieved and the summary still failed.

"Non-empty" is therefore the wrong acceptance test. This module supplies the
missing one: cosine similarity between the query and the text of the rows, using
the same sentence-transformer already loaded for entity linking, so it costs one
short encode per candidate set and needs no extra model or API call.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from sentence_transformers import util

from . import config as _cfg

# A candidate row set whose best row scores below this is treated as "we found
# something, but not the thing that was asked about" -- the cascade keeps
# looking rather than accepting it. Calibrated against the observed separation
# between on-topic and incidental matches (an unrelated fact for the right
# company typically lands ~0.1-0.25; a genuinely responsive fact ~0.4+).
RELEVANCE_FLOOR = _cfg.RELEVANCE_FLOOR


def row_text(row: Any) -> str:
    """Flatten a Neo4j result row into the text a reader would judge it by.

    Mirrors the branches in summarization.format_kg_rows_for_summary so the
    score is computed over the same content the summariser will actually see,
    rather than over fields that never reach the model.
    """
    data = dict(row)
    parts: list[str] = []

    if data.get("fact"):
        parts.append(str(data["fact"]))
    if data.get("attribute") is not None and data.get("value") is not None:
        parts.append(f"{data.get('subject_name', '')} {data['attribute']} {data['value']}")
    # Comparison and count templates carry their evidence in list-valued fields.
    for key in ("facts_a", "facts_b", "facts"):
        value = data.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value if v)
    if not parts:
        # Legacy path/transit modes and generic triples have no fact text.
        triple = [
            str(data.get(k))
            for k in ("subject_name", "relationship", "object_name")
            if data.get(k)
        ]
        if triple:
            parts.append(" ".join(triple))
    return " ".join(p for p in parts if p).strip()


def _mask(text: str, terms: Iterable[str] | None) -> str:
    """Remove the given entity names from `text` (case-insensitive)."""
    if not terms:
        return text
    out = text
    for term in sorted((t for t in terms if t), key=len, reverse=True):
        out = re.sub(re.escape(term), " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def score_rows(
    query: str, rows: list, embedder, mask_terms: Iterable[str] | None = None
) -> list[float]:
    """Cosine similarity of each row's text against the query.

    `mask_terms` (the resolved entity names) are stripped from BOTH sides first,
    and doing so is what makes the score mean anything. Every row returned for a
    query about a company repeats that company's name, and the shared name
    dominates the embedding: "What is the ticker symbol of Karur Vysya Bank?"
    against "Karur Vysya Bank is headquartered in Karur." scores 0.724 with the
    name left in and 0.197 with it removed. Unmasked, the gate waved through
    essentially any fact about the right bank -- which is exactly how a ticker
    lookup ended up answered with the head-office city.

    What is left after masking is the part that actually distinguishes a
    responsive fact from an irrelevant one: the predicate and the topic.
    """
    if not rows:
        return []
    texts = [_mask(row_text(r), mask_terms) for r in rows]
    scorable = [(i, t) for i, t in enumerate(texts) if t]
    if not scorable:
        return [0.0] * len(rows)

    masked_query = _mask(query, mask_terms)
    query_emb = embedder.encode(masked_query, convert_to_tensor=True, normalize_embeddings=True)
    row_emb = embedder.encode(
        [t for _, t in scorable], convert_to_tensor=True, normalize_embeddings=True
    )
    sims = util.cos_sim(query_emb, row_emb)[0]

    scores = [0.0] * len(rows)
    for (idx, _), sim in zip(scorable, sims):
        scores[idx] = float(sim)
    return scores


def best_score(
    query: str, rows: list, embedder, mask_terms: Iterable[str] | None = None
) -> float:
    """Score of the single most on-topic row -- the cascade's acceptance test.

    Deliberately the max, not the mean: a strategy that returns the right fact
    alongside ten irrelevant ones has still answered the question, and averaging
    would reject it.
    """
    scores = score_rows(query, rows, embedder, mask_terms)
    return max(scores) if scores else 0.0


def rank_rows(
    query: str,
    rows: list,
    embedder,
    *,
    limit: int | None = None,
    mask_terms: Iterable[str] | None = None,
) -> list:
    """Reorder rows most-relevant-first, optionally keeping only the top `limit`.

    Used when merging template results with semantic search hits, so the
    summariser reads the strongest evidence first and any cap truncates the
    weakest rows rather than an arbitrary Neo4j ordering.
    """
    if not rows:
        return []
    scores = score_rows(query, rows, embedder, mask_terms)
    ordered = [row for _, row in sorted(zip(scores, rows), key=lambda pair: pair[0], reverse=True)]
    return ordered[:limit] if limit else ordered


def row_key(row: Any) -> tuple:
    """Identity for de-duplicating rows merged from different strategies."""
    data = dict(row)
    return (
        str(data.get("subject_name") or ""),
        str(data.get("relationship") or ""),
        str(data.get("object_name") or ""),
        str(data.get("fact") or ""),
        str(data.get("attribute") or ""),
    )


def merge_rows(primary: list, extra: list, *, cap: int = 40) -> list:
    """Append `extra` rows not already present in `primary`, preserving order."""
    seen = {row_key(r) for r in primary}
    merged = list(primary)
    for row in extra:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        if len(merged) >= cap:
            break
    return merged
