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

from ..paths import DATA_DIR
from ..routing.template_classes import FORMICA_NEEDED_TYPES

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


# Corporate-suffix/prefix noise that's part of a KG company's full legal name but
# almost never appears in a colloquial question ("Karnataka Bank", not "Karnataka
# Bank Limited"). Left unstripped, the full-name substring match never fires, and
# a shorter unrelated node (e.g. the "Karnataka" Geography) that happens to be a
# literal substring of the colloquial name wins the span instead -- silently
# resolving the query to the wrong entity type entirely.
_CORPORATE_SUFFIX_RE = re.compile(
    r"\s+(?:limited|ltd\.?|pvt\.?\s*ltd\.?)\s*$", re.IGNORECASE
)
_CORPORATE_PREFIX_RE = re.compile(r"^\s*the\s+", re.IGNORECASE)


def _name_variants(name: str) -> list[str]:
    """Full name first, then progressively stripped of legal prefix/suffix noise.

    Only accepted when the stripped form still has >= 2 words: "Karnataka Bank
    Limited" -> "Karnataka Bank" is a safe, still-distinctive colloquial name, but
    "HDFC Limited" -> "HDFC" is not -- it collapses to a bare brand prefix that's
    also a substring of a genuinely different company ("HDFC Bank"), so accepting
    it would steal that other company's span instead of just filling a real gap.
    """
    variants = [name]
    stripped = _CORPORATE_SUFFIX_RE.sub("", name)
    stripped = _CORPORATE_PREFIX_RE.sub("", stripped)
    if stripped != name and len(stripped) >= 3 and len(stripped.split()) >= 2:
        variants.append(stripped)
    return variants


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
        for variant in _name_variants(name):
            # Word-boundary matching (same as alias_extract already does) --
            # without it, a short node name is a plain substring search and
            # matches mid-word: "LIC" (the shareholder entity, 3 chars, over
            # the len<3 skip threshold) matched inside "licensed"/"license",
            # hijacking entity resolution for any query using that word
            # (confirmed: "which banks are licensed as universal banks..."
            # and "closest to converting to a universal bank license" both
            # got nnp1/nnp2 wrongly bound to the LIC shareholder node).
            pattern = re.compile(r"\b" + re.escape(variant) + r"\b", flags=re.IGNORECASE)
            matched_here = False
            for m in pattern.finditer(query):
                span = (m.start(), m.end())
                if any(_overlap(span, (s, e)) for s, e, _ in spans):
                    continue
                matched_here = True
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
            if matched_here:
                break
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

    # Very short spans ("CE", "MD", ...) are almost always NER fragmentation noise
    # (e.g. "CET1" split down to just "CE") rather than genuine short entity
    # mentions -- real short names (tickers like "NSE"/"BSE") are already covered
    # by the gazetteer's exact match, so the bar for accepting one *here*, via
    # fuzzy embedding similarity, needs to be much higher than for a full phrase.
    # 0.55 let a "CE" -> "NSE" match at 0.65 through, silently substituting an
    # unrelated stock-exchange entity for what should have been a Metric lookup.
    if len(entity) <= 4 and best_score < 0.85:
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
    matches_df = _drop_generic_indian_banks_false_match(query, matches_df)
    entity_texts = matches_df["entity"].tolist() if not matches_df.empty else []
    return entity_texts, matches_df


# "Across all 39 listed Indian banks in this dataset, which report the
# highest/lowest <metric>?" is this dataset's standard cross-company-ranking
# phrasing, but "Indian banks" (generic, plural, adjective + noun) keeps
# getting NER/semantically linked to the one specific COMPANY literally named
# "Indian Bank" -- confirmed directly: this silently hijacked nnp1/nnp2 for at
# least 6 F_GlobalRank queries (CET1 ratio, branch network size, ...), which
# then tried to rank by that one company's own node instead of the intended
# metric, and always came back empty. Scoped narrowly to this dataset's exact
# recurring phrase (not a blanket "never match Indian Bank" rule, which would
# break the many OTHER queries that genuinely ask about that specific bank).
_GENERIC_INDIAN_BANKS_RE = re.compile(r"\blisted indian banks\b", re.I)


def _drop_generic_indian_banks_false_match(query: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty or not _GENERIC_INDIAN_BANKS_RE.search(query):
        return matches_df
    return matches_df[matches_df["matched_node_name"].astype(str) != "Indian Bank"].reset_index(drop=True)


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
    merged = merge_matches(matches_df.to_dict(orient="records"), to_add)
    # This sweep runs its own independent gazetteer_extract() call, so it can
    # re-add a match resolve_entities() already filtered out (e.g. "Indian
    # Bank" from the generic "listed Indian banks" phrase) -- apply the same
    # filter again here.
    return _drop_generic_indian_banks_false_match(query, merged)


# Backwards-compatible aliases.
enrich_for_template = enrich_for_formica_template
enrich_for_template_with_nodes = enrich_for_formica_template
