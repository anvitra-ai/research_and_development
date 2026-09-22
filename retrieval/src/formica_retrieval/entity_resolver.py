"""Entity extraction and KG linking for financial queries.

Linking strategy (in priority order for overlapping spans):

  1. Gazetteer — exact substring match against KG node labels (longest first).
  2. Aliases — dictionary lookup from kg_entity_aliases.json.
  3. NER + semantic search — HuggingFace NER spans linked via sentence embeddings,
     with optional type boosting when a Formica template is known.

The pipeline calls resolve_entities() twice: once before classification and once
after, passing template_label so semantic search prefers node types the template needs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import util

from formica_template_classes import FORMICA_NEEDED_TYPES
from paths import DATA_DIR

ALIAS_PATH = DATA_DIR / "kg_entity_aliases.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Investors referenced in training data but not always in the KG.
# Block weak semantic matches (e.g. KKR -> JK Tyre).
KNOWN_INVESTOR_PHRASES = {
    "kkr", "temasek", "adia", "gic", "softbank", "mubadala", "qia",
    "cpp investments", "warburg pincus", "warburg pincin",
}

TEMPLATE_NEEDED_TYPES: dict[str, set[str]] = {
    "T_StockImpact": {"MACRO_VAR", "EVENT", "GEOGRAPHY", "COMPANY"},
    "T_CausalChain": {"MACRO_VAR", "EVENT", "GEOGRAPHY", "COMPANY", "SECTOR"},
    "T_CompareExposure": {"MACRO_VAR", "EVENT", "GEOGRAPHY", "COMPANY"},
    "T_MacroTransmission": {"MACRO_VAR", "EVENT", "GEOGRAPHY", "COMPANY"},
    "T_Beneficiary": {"MACRO_VAR", "EVENT", "GEOGRAPHY"},
    "T_EventScenario": {"EVENT", "GEOGRAPHY", "MACRO_VAR"},
    "T_TransitRisk": {"RAW_MATERIAL", "PRODUCT", "GEOGRAPHY"},
    "T_InvestmentFlow": {"INVESTOR", "SECTOR", "COMPANY", "GEOGRAPHY"},
    "T_SupplyDisruption": {"RAW_MATERIAL", "PRODUCT", "GEOGRAPHY", "COMPANY"},
    "T_Hedging": {"MACRO_VAR", "EVENT", "GEOGRAPHY", "COMPANY"},
    "T_PriceLinkage": {"MACRO_VAR", "RAW_MATERIAL", "COMPANY"},
}


# ---------------------------------------------------------------------------
# Alias / gazetteer extraction
# ---------------------------------------------------------------------------


def load_aliases(path: str | Path = ALIAS_PATH) -> dict[str, dict[str, str]]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------


def texts_from_entities(ents: list) -> list[dict[str, Any]]:
    """Return NER spans with character offsets when available."""
    out: list[dict[str, Any]] = []
    for e in ents:
        if isinstance(e, dict):
            text = (e.get("word") or e.get("text") or "").strip()
            if not text:
                continue
            start = e.get("start")
            end = e.get("end")
            out.append({"text": text, "start": start, "end": end})
        else:
            out.append({"text": str(e).strip(), "start": None, "end": None})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        key = item["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _find_span(query: str, phrase: str) -> tuple[int, int] | None:
    pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    match = pattern.search(query)
    if match:
        return match.start(), match.end()
    return None


def gazetteer_extract(query: str, node_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find KG node names mentioned verbatim in the query (longest match first)."""
    names = (
        node_df.dropna(subset=["node_name"])
        .drop_duplicates(subset=["node_name"])
        .sort_values("node_name", key=lambda s: s.str.len(), ascending=False)
    )
    spans: list[tuple[int, int, dict[str, Any]]] = []
    for _, row in names.iterrows():
        name = str(row["node_name"])
        if len(name) < 3:
            continue
        pattern = re.compile(re.escape(name), flags=re.IGNORECASE)
        for m in pattern.finditer(query):
            span = (m.start(), m.end())
            if any(_overlap(span, (s, e)) for s, e, _ in spans):
                continue
            spans.append(
                (
                    span[0],
                    span[1],
                    {
                        "entity": m.group(0),
                        "matched_node_name": name,
                        "matched_id": row["id"],
                        "matched_type": row["node_type"],
                        "similarity": 1.0,
                        "source": "gazetteer",
                    },
                )
            )
    spans.sort(key=lambda x: x[0])
    return [item for _, _, item in spans]


def alias_extract(query: str, aliases: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Match alias phrases in the query (longest phrase first)."""
    hits: list[tuple[int, int, dict[str, Any]]] = []
    for phrase, meta in sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        for m in pattern.finditer(query):
            span = (m.start(), m.end())
            if any(_overlap(span, (s, e)) for s, e, _ in hits):
                continue
            hits.append(
                (
                    span[0],
                    span[1],
                    {
                        "entity": m.group(0),
                        "matched_node_name": meta["name"],
                        "matched_id": meta["id"],
                        "matched_type": meta["type"],
                        "similarity": 0.99,
                        "source": "alias",
                    },
                )
            )
    hits.sort(key=lambda x: x[0])
    return [item for _, _, item in hits]


def _occupied_spans(query: str, *groups: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for group in groups:
        for item in group:
            found = _find_span(query, item["entity"])
            if found:
                spans.append(found)
    return spans


def _is_blocked_ner_span(
    query: str,
    ner_item: dict[str, Any],
    occupied: list[tuple[int, int]],
) -> bool:
    text = ner_item["text"]
    start = ner_item.get("start")
    end = ner_item.get("end")

    if start is not None and end is not None:
        span = (int(start), int(end))
        if any(_overlap(span, occ) for occ in occupied):
            return True

    found = _find_span(query, text)
    if found and any(_overlap(found, occ) for occ in occupied):
        return True

    # Drop NER fragments inside a longer gazetteer/alias span.
    if found:
        for occ in occupied:
            if found[0] >= occ[0] and found[1] <= occ[1] and (found[0], found[1]) != occ:
                return True

    return False


def _semantic_link(
    entity: str,
    embedder,
    node_df: pd.DataFrame,
    node_emb,
    preferred_types: set[str] | None = None,
    top_k: int = 15,
) -> dict[str, Any] | None:
    if entity.lower() in KNOWN_INVESTOR_PHRASES:
        investor_rows = node_df[node_df["node_type"] == "INVESTOR"]
        for _, row in investor_rows.iterrows():
            if entity.lower() in str(row["node_name"]).lower():
                return {
                    "entity": entity,
                    "matched_node_name": row["node_name"],
                    "matched_id": row["id"],
                    "matched_type": row["node_type"],
                    "similarity": 0.95,
                    "source": "semantic",
                }
        return None

    entity_emb = embedder.encode(entity, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(entity_emb, node_emb, top_k=top_k)[0]

    best_score = -1.0
    best_row = None
    for hit in hits:
        row = node_df.iloc[hit["corpus_id"]]
        score = float(hit["score"])
        if row["node_name"].lower() == entity.lower():
            score += 0.35
        if preferred_types and row["node_type"] in preferred_types:
            score += 0.2
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None

    if best_score < 0.55 and len(entity) <= 4:
        return None

    return {
        "entity": entity,
        "matched_node_name": best_row["node_name"],
        "matched_id": best_row["id"],
        "matched_type": best_row["node_type"],
        "similarity": best_score,
        "source": "semantic",
    }


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def merge_matches(*groups: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge entity hits, preferring gazetteer/alias over semantic for same span."""
    priority = {"gazetteer": 3, "alias": 2, "semantic": 1}
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            key = item["matched_id"]
            existing = by_id.get(key)
            if existing is None or priority.get(item.get("source", ""), 0) > priority.get(
                existing.get("source", ""), 0
            ):
                by_id[key] = item
    if not by_id:
        return pd.DataFrame(
            columns=["entity", "matched_node_name", "matched_id", "matched_type", "similarity"]
        )
    df = pd.DataFrame(by_id.values())
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


def parse_type_hints(type_spec: str | None) -> set[str]:
    if not type_spec:
        return set()
    return {t.strip() for t in type_spec.split("|") if t.strip()}


def resolve_entities(
    query: str,
    ner_pipeline,
    embedder,
    node_df: pd.DataFrame,
    node_emb,
    *,
    aliases: dict[str, dict[str, str]] | None = None,
    template_label: str | None = None,
    source_type: str | None = None,
    target_type: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Extract and link entities using gazetteer, aliases, NER, and semantic search."""
    aliases = aliases or load_aliases()

    # High-confidence exact matches first; their spans block overlapping NER hits.
    gazetteer_hits = gazetteer_extract(query, node_df)
    alias_hits = alias_extract(query, aliases)
    occupied = _occupied_spans(query, gazetteer_hits, alias_hits)

    # On the second pass, bias semantic search toward types the template class expects.
    preferred = None
    if template_label:
        preferred = parse_type_hints(source_type) | parse_type_hints(target_type)
        preferred |= TEMPLATE_NEEDED_TYPES.get(template_label, set())
        preferred |= FORMICA_NEEDED_TYPES.get(template_label, set())

    ner_hits = []
    for ner_item in texts_from_entities(ner_pipeline(query)):
        text = ner_item["text"]
        if any(text.lower() == g["entity"].lower() for g in gazetteer_hits + alias_hits):
            continue
        if _is_blocked_ner_span(query, ner_item, occupied):
            continue
        linked = _semantic_link(
            text, embedder, node_df, node_emb, preferred_types=preferred or None
        )
        if linked:
            ner_hits.append(linked)

    matches_df = merge_matches(gazetteer_hits, alias_hits, ner_hits)
    entity_texts = matches_df["entity"].tolist() if not matches_df.empty else []
    return entity_texts, matches_df


# ---------------------------------------------------------------------------
# Post-classification enrichment
# ---------------------------------------------------------------------------


def enrich_for_formica_template(
    query: str,
    template_label: str,
    matches_df: pd.DataFrame,
    node_df: pd.DataFrame,
    aliases: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Post-classification entity enrichment for Formica templates.

    After classification we know which node types the template needs (e.g. GEOGRAPHY
    for transit queries). This sweep adds any gazetteer/alias hits of those types
    that the first two resolution passes missed.
    """
    aliases = aliases or load_aliases()
    needed_types = FORMICA_NEEDED_TYPES.get(template_label, set())
    extra = alias_extract(query, aliases) + gazetteer_extract(query, node_df)
    existing_ids = set(matches_df["matched_id"].tolist()) if not matches_df.empty else set()
    to_add = [
        hit
        for hit in extra
        if hit["matched_id"] not in existing_ids
        and (not needed_types or hit["matched_type"] in needed_types)
    ]
    if not to_add:
        return matches_df
    return merge_matches(matches_df.to_dict(orient="records"), to_add)


# Backwards-compatible aliases.
enrich_for_template = enrich_for_formica_template
enrich_for_template_with_nodes = enrich_for_formica_template
