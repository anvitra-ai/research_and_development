"""Instantiate Formica template classes with triplet slots and run Cypher.

Formica templates use *triplet slots* — placeholders filled from resolved entities:

  nnp1, nnp2, nnp3  — KG node IDs (named entity positions)
  nn1, nn2          — node types for optional type guards in Cypher
  prop1             — relationship type (TRANSITS, BENEFITS_FROM, CAUSES, …)
  threshold         — numeric bound for quantitative templates

expand_formica_template() picks slots from the query + entities, selects either
the default template Cypher or a specialized mode (transit, propagation, etc.),
then execute_formica_template() runs it against Neo4j with a chain of fallbacks
when the primary query returns no rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from entity_resolver import enrich_for_formica_template  # noqa: F401 — re-exported for notebooks
from formica_template_classes import (
    RELATION_KEYWORDS,
    SLOT_SHOCK_PRIORITY,
    SLOT_SUBJECT_PRIORITY,
)
from formica_template_classifier import extract_threshold
from paths import DATA_DIR

DEFAULT_TEMPLATE_PATH = DATA_DIR / "formica_query_templates.json"

# Regex hints that steer slot assignment before the generic F_Simple triplet is built.
# e.g. "transit" -> prop1=TRANSITS, "benefit" -> prop1=BENEFITS_FROM
_QUERY_HINTS: list[tuple[re.Pattern[str], dict[str, Any]]] = [
    (re.compile(r"\btransit\b|\bpass(?:es)? through\b", re.I), {"prop1": "TRANSITS", "subject_types": ["RAW_MATERIAL", "PRODUCT", "GEOGRAPHY"]}),
    (re.compile(r"\bbeneficiar|\bwinner|\bwins from\b", re.I), {"prop1": "BENEFITS_FROM", "subject_types": ["SECTOR", "COMPANY"]}),
    (re.compile(r"\bown(?:s|ership)?\b|\bsubsidiar", re.I), {"prop1": "OWNS", "subject_types": ["COMPANY", "INVESTOR"]}),
    (re.compile(r"\bhedg", re.I), {"prop1": "HEDGES", "subject_types": ["COMPANY"]}),
    (re.compile(r"\bregulat|\bpolicy\b|\bgovernment intervention\b", re.I), {"prop1": "REGULATED_BY", "subject_types": ["COMPANY", "SECTOR"]}),
    (re.compile(r"\bconstrain|\bbottleneck|\blimit\b", re.I), {"prop1": "CONSTRAINED_BY", "subject_types": ["COMPANY", "SECTOR"]}),
    (re.compile(r"\bconstituent|\bcompanies in\b|\bin sector\b", re.I), {"prop1": "CONTAINS", "subject_types": ["SECTOR"]}),
    (re.compile(r"\bcausal|lead(?:s)? to|propagat|transmission channel|flow into|dependence\b", re.I), {"prop1": "CAUSES", "subject_types": ["EVENT", "MACRO_VAR", "GEOGRAPHY", "RAW_MATERIAL"]}),
    (re.compile(r"\bexpos|\baffect|\bimpact|\bhurt\b", re.I), {"prop1": "HURT_BY", "subject_types": ["COMPANY", "SECTOR"]}),
    (re.compile(r"\binvest", re.I), {"prop1": "INVESTS_IN", "subject_types": ["INVESTOR"]}),
    (re.compile(r"\bprice link|linked to|track\b", re.I), {"prop1": "PRICE_LINKED_TO", "subject_types": ["COMPANY", "MACRO_VAR"]}),
    (re.compile(r"\bsupply|sourced from|feedstock\b", re.I), {"prop1": "SOURCED_FROM", "subject_types": ["COMPANY", "RAW_MATERIAL", "PRODUCT"]}),
]

# Best-effort numeric value extraction from a Graphiti edge's flattened attributes
# (property name varies by which edge_type class populated it -- ExposureRelation,
# OwnershipStake, MetricObservation, etc. -- so try every plausible one in order).
_NUMERIC_COALESCE = (
    "coalesce(r.percentage, r.exposure_percentage, r.exposure_amount, "
    "r.ownership_percentage, r.value, r.share, r.weight, 1.0)"
)

# Alternative Cypher patterns for query shapes the generic triplet cannot express.
_SPECIAL_SIMPLE_CYPHER: dict[str, str] = {
    "sector_beneficiaries": (
        "MATCH (sector:Node {id: $nnp1})-[:CONTAINS]->(company:Node {type: 'COMPANY'})"
        "-[:BENEFITS_FROM]->(shock:Node)\n"
        "WHERE ($nnp2 IS NULL OR shock.id = $nnp2)\n"
        "RETURN sector.label AS sector_name, company.label AS company_name, "
        "shock.label AS shock_name, 'BENEFITS_FROM' AS relationship"
    ),
    "shock_beneficiaries": (
        "MATCH (company:Node {type: 'COMPANY'})-[r:BENEFITS_FROM]->(shock:Node {id: $nnp1})\n"
        "RETURN company.label AS company_name, shock.label AS shock_name, type(r) AS relationship\n"
        "ORDER BY company_name"
    ),
    "product_producers": (
        "MATCH (company:Node {type: 'COMPANY'})-[r:PRODUCES]->(product:Node {id: $nnp1})\n"
        "RETURN company.label AS company_name, product.label AS product_name, type(r) AS relationship\n"
        "ORDER BY company_name"
    ),
    "beneficiary_paths": (
        "MATCH p = (source:Node {id: $nnp1})-[*1..6]-(target:Node {type: 'COMPANY'})\n"
        "WHERE source.id <> target.id\n"
        "RETURN source.label AS source_name, target.label AS target_name, "
        "[n IN nodes(p) | n.label] AS path_names, "
        "[r IN relationships(p) | type(r)] AS rel_types, length(p) AS hops\n"
        "ORDER BY length(p)\n"
        "LIMIT 25"
    ),
    "sector_constituents": (
        "MATCH (sector:Node {id: $nnp1})-[:CONTAINS]->(company:Node {type: 'COMPANY'})\n"
        "RETURN sector.label AS sector_name, company.label AS company_name, company.type AS node_type"
    ),
    "commodity_transit": (
        "MATCH (commodity:Node {id: $nnp1})-[r:TRANSITS]->(geo:Node)\n"
        "WHERE ($nnp2 IS NULL OR geo.id = $nnp2)\n"
        "RETURN commodity.label AS commodity_name, geo.label AS geography_name, "
        "type(r) AS relationship, r.share AS share, r.channel AS channel"
    ),
    "transit_quant": (
        "MATCH (commodity:Node {id: $nnp1})-[r:TRANSITS]->(geo:Node)\n"
        "WHERE ($nnp2 IS NULL OR geo.id = $nnp2)\n"
        "RETURN commodity.label AS commodity_name, geo.label AS geography_name, "
        "r.share AS share, r.channel AS channel, type(r) AS relationship"
    ),
    "path_trace": (
        "MATCH p = shortestPath((source:Node {id: $nnp1})-[*..8]-(target:Node {id: $nnp2}))\n"
        "RETURN source.label AS source_name, target.label AS target_name, "
        "[n IN nodes(p) | n.label] AS path_names, "
        "[r IN relationships(p) | type(r)] AS rel_types, length(p) AS hops"
    ),
    "propagation": (
        "MATCH p = shortestPath((source:Node {id: $nnp1})-[*..8]-(target:Node {id: $nnp2}))\n"
        "RETURN source.label AS source_name, target.label AS target_name, "
        "[n IN nodes(p) | n.label] AS path_names, "
        "[r IN relationships(p) | type(r)] AS rel_types, length(p) AS hops"
    ),
    "propagation_open": (
        "MATCH p = (source:Node {id: $nnp1})-[*1..8]-(target:Node)\n"
        "WHERE target.type IN ['COMPANY', 'MACRO_VAR', 'SECTOR', 'EVENT', 'GEOGRAPHY']\n"
        "  AND source.id <> target.id\n"
        "RETURN source.label AS source_name, target.label AS target_name, "
        "[n IN nodes(p) | n.label] AS path_names, "
        "[r IN relationships(p) | type(r)] AS rel_types, length(p) AS hops\n"
        "ORDER BY length(p)\n"
        "LIMIT 25"
    ),
    "propagation_all_paths": (
        "MATCH p = (source:Node {id: $nnp1})-[*1..8]-(target:Node {id: $nnp2})\n"
        "RETURN source.label AS source_name, target.label AS target_name, "
        "[n IN nodes(p) | n.label] AS path_names, "
        "[r IN relationships(p) | type(r)] AS rel_types, length(p) AS hops\n"
        "ORDER BY length(p)\n"
        "LIMIT 10"
    ),
    # "Across all 39 banks, which reports the highest/lowest <metric>?" -- rank
    # every entity of nn1's type (e.g. Company) connected to the specific resolved
    # metric entity (nnp2), unlike the generic F_QuantMax/Min templates which rank
    # one subject's own neighbors. Graphiti schema (:Entity/:RELATES_TO/r.fact),
    # not the legacy :Node schema the rest of this dict uses.
    "global_ranking_max": (
        "MATCH (s:Entity)-[r:RELATES_TO]-(o:Entity {uuid: $nnp2})\n"
        "WHERE ($nn1 IS NULL OR $nn1 IN [l IN labels(s) | toUpper(l)])\n"
        f"WITH s, o, r, {_NUMERIC_COALESCE} AS val\n"
        "ORDER BY val DESC\n"
        "RETURN s.name AS subject_name, o.name AS object_name, val AS max_value, r.fact AS fact\n"
        "LIMIT 1"
    ),
    "global_ranking_min": (
        "MATCH (s:Entity)-[r:RELATES_TO]-(o:Entity {uuid: $nnp2})\n"
        "WHERE ($nn1 IS NULL OR $nn1 IN [l IN labels(s) | toUpper(l)])\n"
        f"WITH s, o, r, {_NUMERIC_COALESCE} AS val\n"
        "ORDER BY val ASC\n"
        "RETURN s.name AS subject_name, o.name AS object_name, val AS min_value, r.fact AS fact\n"
        "LIMIT 1"
    ),
}

_PROPAGATION_RE = re.compile(
    r"\bpropagat|flow into|transmission channel|lead(?:s)? to|dependence\b",
    re.I,
)
_TRANSIT_QUANT_RE = re.compile(r"\btransit|share of|how much\b.*\btransit", re.I)
_BENEFICIARY_RE = re.compile(
    r"\bbeneficiar|\bwinner|\bbenefit from\b|\bwill benefit\b|\bwho benefits\b|\bcompanies will benefit\b",
    re.I,
)
_SHOCK_TYPES = frozenset(
    {"EVENT", "MACRO_VAR", "PRODUCT", "RAW_MATERIAL", "TECHNOLOGY", "CONSTRAINT", "GEOGRAPHY", "POLICY"}
)
_BENEFICIARY_FALLBACKS = ("shock_beneficiaries", "beneficiary_paths", "product_producers")

_COMPARE_NO_SHOCK_CYPHER = {
    "F_CompMore": (
        "MATCH (a:Entity {uuid: $nnp1}), (b:Entity {uuid: $nnp2})\n"
        "WHERE a.uuid <> b.uuid\n"
        "OPTIONAL MATCH pa = shortestPath((a)-[*..8]-(b))\n"
        "RETURN a.name AS entity_a, b.name AS entity_b, length(pa) AS hops_a, "
        "CASE WHEN length(pa) > 0 THEN a.name ELSE b.name END AS more_connected, "
        "[rel IN relationships(pa) | rel.fact] AS facts"
    ),
    "F_CompLess": (
        "MATCH (a:Entity {uuid: $nnp1}), (b:Entity {uuid: $nnp2})\n"
        "WHERE a.uuid <> b.uuid\n"
        "OPTIONAL MATCH pa = shortestPath((a)-[*..8]-(b))\n"
        "RETURN a.name AS entity_a, b.name AS entity_b, length(pa) AS hops, "
        "CASE WHEN length(pa) > 0 THEN b.name ELSE a.name END AS less_connected, "
        "[rel IN relationships(pa) | rel.fact] AS facts"
    ),
    "F_CompApprox": (
        "MATCH (a:Entity {uuid: $nnp1}), (b:Entity {uuid: $nnp2})\n"
        "WHERE a.uuid <> b.uuid\n"
        "OPTIONAL MATCH pa = shortestPath((a)-[*..8]-(b))\n"
        "WITH a, b, length(pa) AS hops, pa\n"
        "WHERE hops <= 2\n"
        "RETURN a.name AS entity_a, b.name AS entity_b, hops, "
        "[rel IN relationships(pa) | rel.fact] AS facts"
    ),
}


def load_templates(path: str | Path = DEFAULT_TEMPLATE_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _query_hints(query: str) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for pattern, values in _QUERY_HINTS:
        if pattern.search(query):
            for key, value in values.items():
                hints.setdefault(key, value)
    return hints


def _entity_position(query: str, entity_text: str) -> int:
    match = re.search(re.escape(entity_text), query, flags=re.IGNORECASE)
    return match.start() if match else 10_000


def _is_parenthetical(query: str, entity_text: str) -> bool:
    """Whether entity_text's span sits inside a "(...)" in the query.

    English regularly restates a vague/broad term with a specific one in
    parentheses right after it -- "asset quality (lowest gross NPA)" means the
    gross NPA *metric specifically*, not the more general "asset quality" -- so
    when two same-type candidates are otherwise tied, the parenthetical one is
    almost always the more precise, intended referent.
    """
    match = re.search(re.escape(entity_text), query, flags=re.IGNORECASE)
    if not match:
        return False
    start, end = match.start(), match.end()
    open_paren = query.rfind("(", 0, start)
    close_paren = query.find(")", end)
    # A "(" before with no ")" in between, and a matching ")" after.
    return open_paren != -1 and close_paren != -1 and query.find(")", open_paren, start) == -1


def _row_type(matched: pd.DataFrame, entity_id: str | None) -> str | None:
    if not entity_id or matched.empty:
        return None
    hits = matched[matched["matched_id"] == entity_id]
    if hits.empty:
        return None
    return hits.iloc[0]["matched_type"]


def _ordered_by_query(query: str, matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched
    positions = [
        _entity_position(query, str(row["entity"]))
        for _, row in matched.iterrows()
    ]
    out = matched.copy()
    out["_pos"] = positions
    return out.sort_values("_pos").drop(columns="_pos").reset_index(drop=True)


def _pick_by_type(
    matched: pd.DataFrame,
    types: set[str] | list[str],
    *,
    exclude_ids: set[str] | None = None,
    ordered: pd.DataFrame | None = None,
) -> str | None:
    exclude_ids = exclude_ids or set()
    type_set = set(types)
    rows = matched[
        matched["matched_type"].isin(type_set) & ~matched["matched_id"].isin(exclude_ids)
    ]
    if rows.empty:
        return None
    if ordered is not None:
        for entity_id in ordered["matched_id"]:
            if entity_id in set(rows["matched_id"]) and entity_id not in exclude_ids:
                return entity_id
    return rows.iloc[0]["matched_id"]


def _pick_distinct(
    matched: pd.DataFrame,
    types: set[str] | list[str],
    exclude_id: str | None,
    *,
    ordered: pd.DataFrame | None = None,
) -> str | None:
    exclude = {exclude_id} if exclude_id else set()
    return _pick_by_type(matched, types, exclude_ids=exclude, ordered=ordered)


def _pick_by_priority(
    matched: pd.DataFrame,
    priorities: list[str],
    *,
    exclude_ids: set[str] | None = None,
    ordered: pd.DataFrame | None = None,
) -> str | None:
    exclude_ids = exclude_ids or set()
    for node_type in priorities:
        picked = _pick_by_type(matched, {node_type}, exclude_ids=exclude_ids, ordered=ordered)
        if picked:
            return picked
    return None


def _params_for_cypher(cypher: str, slots: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bind every $param referenced in Cypher from slots/overrides (None when optional)."""
    needed = set(re.findall(r"\$(\w+)", cypher))
    overrides = overrides or {}
    params: dict[str, Any] = {}
    for key in needed:
        if key in overrides:
            params[key] = overrides[key]
        elif slots.get(key) is not None:
            params[key] = slots[key]
        else:
            params[key] = None
    return params


def _parse_propagate_to(query: str, matched: pd.DataFrame) -> str | None:
    match = re.search(r"\bpropagat(?:e|ion)?\s+to\s+(.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if not match:
        return None
    phrase = match.group(1).strip()
    ordered = _ordered_by_query(query, matched)
    for _, row in ordered.iterrows():
        entity = str(row["entity"])
        if entity.lower() in phrase.lower() or phrase.lower() in entity.lower():
            return row["matched_id"]
    company_id = _pick_by_type(matched, {"COMPANY"}, ordered=ordered)
    return company_id


def _parse_from_to(query: str, matched: pd.DataFrame) -> tuple[str | None, str | None]:
    """Extract source/target IDs from 'from X ... to Y' phrasing."""
    ordered = _ordered_by_query(query, matched)
    if len(ordered) < 2:
        return None, None
    lower = query.lower()
    from_idx = lower.find(" from ")
    to_idx = lower.find(" to ")
    if from_idx == -1 or to_idx == -1 or to_idx <= from_idx:
        return None, None
    first = ordered.iloc[0]["matched_id"]
    second = ordered.iloc[1]["matched_id"]
    return first, second


def _pick_subject(
    query: str,
    template_label: str,
    matched: pd.DataFrame,
    hints: dict[str, Any],
) -> tuple[str | None, str | None]:
    ordered = _ordered_by_query(query, matched)
    priorities = hints.get("subject_types") or SLOT_SUBJECT_PRIORITY.get(
        template_label, SLOT_SUBJECT_PRIORITY["F_Simple"]
    )

    from_id, to_id = _parse_from_to(query, matched)
    if from_id:
        return from_id, _row_type(matched, from_id)

    # "affect/impact X" — X is often the grammatical object but valid KG anchor.
    affect_match = re.search(
        r"\b(?:affect|impact|expose|hurt|benefit|constrain|regulate)\w*\s+(.+?)(?:\?|$)",
        query,
        flags=re.IGNORECASE,
    )
    if affect_match:
        phrase = affect_match.group(1).strip()
        for _, row in ordered.iterrows():
            if str(row["entity"]).lower() in phrase.lower():
                return row["matched_id"], row["matched_type"]

    subject_id = _pick_by_priority(matched, priorities, ordered=ordered)
    if subject_id is None and not ordered.empty:
        subject_id = ordered.iloc[0]["matched_id"]
    return subject_id, _row_type(matched, subject_id)


def _pick_shock_id(matched: pd.DataFrame, exclude_ids: set[str] | None = None) -> str | None:
    return _pick_by_priority(matched, SLOT_SHOCK_PRIORITY, exclude_ids=exclude_ids)


def infer_prop1(query: str, template_label: str, templates: dict, hints: dict[str, Any] | None = None) -> str | None:
    hints = hints or _query_hints(query)
    if hints.get("prop1"):
        return hints["prop1"]

    q = query.lower()
    # Longer / more specific keywords first.
    ranked = sorted(
        RELATION_KEYWORDS.items(),
        key=lambda item: max(len(kw) for kw in item[1]),
        reverse=True,
    )
    for rel, keywords in ranked:
        if any(kw in q for kw in keywords):
            return rel

    spec = templates["templates"].get(template_label, {})
    default = spec.get("default_relationships")
    if default:
        return default.split("|")[0]
    return None


def _dedupe_slots(
    nnp1: str | None, nnp2: str | None, matched: pd.DataFrame, ordered: pd.DataFrame, query: str = ""
) -> str | None:
    if not nnp2 or nnp2 == nnp1:
        # When falling back to "any other distinct entity" (no type filter narrowed
        # it down), prefer a parenthetical mention over leftmost position -- "asset
        # quality (lowest gross NPA)" means the gross NPA metric specifically, and
        # this fallback is exactly the path that otherwise silently picks whichever
        # candidate happens to appear first in the raw sentence. Only overrides when
        # there's exactly one parenthetical candidate; ambiguous cases (zero or
        # several) fall through to the existing position-based pick unchanged.
        candidates = matched[matched["matched_id"] != nnp1] if nnp1 else matched
        parenthetical_ids = [
            row["matched_id"]
            for _, row in candidates.iterrows()
            if _is_parenthetical(query, str(row["entity"]))
        ]
        if len(parenthetical_ids) == 1:
            return parenthetical_ids[0]
        return _pick_distinct(matched, set(matched["matched_type"]), nnp1, ordered=ordered)
    return nnp2


def _special_simple_mode(query: str, slots: dict[str, Any], matched: pd.DataFrame | None = None) -> str | None:
    q = query.lower()
    nnp1_type = slots.get("nn1")
    if _PROPAGATION_RE.search(query) and slots.get("nnp1") and slots.get("nnp2"):
        return "propagation"
    if _PROPAGATION_RE.search(query) and slots.get("nnp1"):
        if matched is not None:
            target_id = _parse_propagate_to(query, matched)
            if target_id and target_id != slots.get("nnp1"):
                slots["nnp2"] = target_id
                slots["nn2"] = _row_type(matched, target_id)
                return "propagation"
        return "propagation_open"
    if re.search(r"\btrace\b|\bpath from\b", q) and slots.get("nnp1") and slots.get("nnp2"):
        return "path_trace"
    if _BENEFICIARY_RE.search(q) and slots.get("nnp1"):
        if nnp1_type == "SECTOR":
            return "sector_beneficiaries"
        if nnp1_type in _SHOCK_TYPES:
            return "shock_beneficiaries"
    if nnp1_type == "SECTOR" and re.search(r"\bconstituent|\bcompanies in\b|\bin\b", q):
        return "sector_constituents"
    if nnp1_type in {"RAW_MATERIAL", "PRODUCT"} and (
        slots.get("prop1") == "TRANSITS" or re.search(r"\btransit", q)
    ):
        return "commodity_transit"
    return None


_GLOBAL_RANKING_RE = re.compile(
    r"\bacross all\b|\bwhich\b(?:\s+\w+){0,3}\s+(?:bank|banks|company|companies)\b", re.I
)


def _quant_mode(query: str, template_label: str, slots: dict[str, Any]) -> str | None:
    if not template_label.startswith("F_Quant"):
        return None
    if slots.get("prop1") == "TRANSITS" or _TRANSIT_QUANT_RE.search(query):
        if slots.get("nn1") in {"RAW_MATERIAL", "PRODUCT", None}:
            return "transit_quant"
    # "Across all 39 listed Indian banks, which report the highest/lowest <metric>?"
    # needs to rank across every entity of nn1's type connected to the resolved
    # metric (nnp2), not one entity's own neighbors -- the generic F_QuantMax/Min
    # Cypher (rank one subject's neighbors) is the wrong shape for this regardless
    # of what nnp1 got picked as (often a stray, wrong entity for these queries).
    if template_label in {"F_QuantMax", "F_QuantMin"} and slots.get("nnp2") and _GLOBAL_RANKING_RE.search(query):
        return "global_ranking_max" if template_label == "F_QuantMax" else "global_ranking_min"
    return None


def extract_triplet_slots(
    query: str,
    template_label: str,
    matches_df: pd.DataFrame,
    *,
    path: str | Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Map resolved entities to Formica slots nnp1, nnp2, prop1, nn1, nn2.

    Slot assignment is template-specific: compare templates pick two companies,
    quant templates pair commodity + geography, propagation queries set source/target,
    and F_Simple uses query hints + entity-type priority to pick subject and predicate.
    """
    templates = load_templates(path)
    matched = (
        matches_df.dropna(subset=["matched_id"])
        .drop_duplicates(subset=["matched_id"])
        .reset_index(drop=True)
    )
    if matched.empty:
        raise ValueError("matches_df has no resolved entities")

    hints = _query_hints(query)
    ordered = _ordered_by_query(query, matched)

    company_ids = matched[matched["matched_type"] == "COMPANY"]["matched_id"].tolist()
    geo_id = _pick_by_type(matched, {"GEOGRAPHY"}, ordered=ordered)
    event_id = _pick_by_type(matched, {"EVENT"}, ordered=ordered)
    macro_id = _pick_by_type(matched, {"MACRO_VAR"}, ordered=ordered)
    sector_id = _pick_by_type(matched, {"SECTOR"}, ordered=ordered)
    commodity_id = _pick_by_type(matched, {"RAW_MATERIAL", "PRODUCT"}, ordered=ordered)
    shock_id = _pick_shock_id(matched) or event_id or macro_id or geo_id

    nnp1, nn1 = _pick_subject(query, template_label, matched, hints)
    nnp2 = None
    nn2 = None
    nnp3 = None

    from_id, to_id = _parse_from_to(query, matched)

    if template_label in {"F_CompMore", "F_CompLess", "F_CompApprox"}:
        nnp1 = company_ids[0] if company_ids else nnp1
        nnp2 = company_ids[1] if len(company_ids) > 1 else _pick_distinct(
            matched, {"COMPANY", "SECTOR"}, nnp1, ordered=ordered
        )
        nnp3 = _pick_shock_id(matched, exclude_ids={nnp1, nnp2} if nnp1 and nnp2 else set())
        if not nnp3:
            nnp3 = shock_id
        nn1 = _row_type(matched, nnp1)
        nn2 = _row_type(matched, nnp2)
    elif template_label in {"F_LogIntersection", "F_LogDifference"}:
        nnp1 = _pick_by_priority(
            matched, ["COMPANY", "SECTOR", "GEOGRAPHY"], ordered=ordered
        ) or nnp1
        nnp2 = _pick_distinct(
            matched, {"COMPANY", "GEOGRAPHY", "EVENT", "MACRO_VAR", "SECTOR"}, nnp1, ordered=ordered
        )
        if from_id and to_id:
            nnp1, nnp2 = from_id, to_id
        nn1 = _row_type(matched, nnp1)
        nn2 = _row_type(matched, nnp2)
    elif template_label.startswith("F_Quant") or template_label.startswith("F_CompCount"):
        if commodity_id and (_TRANSIT_QUANT_RE.search(query) or hints.get("prop1") == "TRANSITS"):
            nnp1 = commodity_id
            nnp2 = geo_id or _pick_distinct(matched, {"GEOGRAPHY"}, commodity_id, ordered=ordered)
        elif template_label == "F_QuantCount" and commodity_id:
            nnp1 = commodity_id
            nnp2 = geo_id
        else:
            nnp1 = _pick_by_priority(
                matched,
                SLOT_SUBJECT_PRIORITY.get(template_label, ["COMPANY", "SECTOR", "GEOGRAPHY"]),
                ordered=ordered,
            ) or nnp1
            nnp2 = _pick_distinct(matched, {"GEOGRAPHY", "COMPANY", "MACRO_VAR"}, nnp1, ordered=ordered)
        nn1 = _row_type(matched, nnp1)
        nn2 = _row_type(matched, nnp2)
        if _TRANSIT_QUANT_RE.search(query) or hints.get("prop1") == "TRANSITS":
            prop1 = "TRANSITS"
    elif template_label == "F_LogUnion":
        nnp1, nn1 = _pick_subject(query, template_label, matched, hints)
        nnp2 = _pick_distinct(matched, set(matched["matched_type"]), nnp1, ordered=ordered)
        nn2 = _row_type(matched, nnp2)
    elif template_label == "F_Simple":
        if _PROPAGATION_RE.search(query):
            source_id = _pick_by_priority(
                matched, ["EVENT", "MACRO_VAR", "GEOGRAPHY", "RAW_MATERIAL", "PRODUCT"], ordered=ordered
            ) or nnp1
            target_id = (
                _parse_propagate_to(query, matched)
                or (company_ids[0] if company_ids else None)
                or _pick_distinct(matched, {"COMPANY", "SECTOR", "MACRO_VAR"}, source_id, ordered=ordered)
            )
            nnp1, nn1 = source_id, _row_type(matched, source_id)
            nnp2, nn2 = target_id, _row_type(matched, target_id)
        elif commodity_id and hints.get("prop1") == "TRANSITS":
            nnp1, nn1 = commodity_id, _row_type(matched, commodity_id)
            nnp2 = geo_id
            nn2 = _row_type(matched, nnp2)
        elif sector_id and re.search(r"\bbeneficiar|\bconstituent|\bcompanies in\b", query, re.I):
            nnp1, nn1 = sector_id, "SECTOR"
            nnp2 = shock_id
            nn2 = _row_type(matched, nnp2)
        elif from_id and to_id:
            nnp1, nnp2 = from_id, to_id
            nn1 = _row_type(matched, nnp1)
            nn2 = _row_type(matched, nnp2)
        else:
            nnp2 = _pick_distinct(
                matched,
                {"EVENT", "MACRO_VAR", "GEOGRAPHY", "RAW_MATERIAL", "PRODUCT", "COMPANY", "SECTOR"},
                nnp1,
                ordered=ordered,
            )
            nn2 = _row_type(matched, nnp2)
    else:
        nnp3 = shock_id

    nnp2 = _dedupe_slots(nnp1, nnp2, matched, ordered, query)
    nn2 = _row_type(matched, nnp2)

    prop1 = locals().get("prop1") or infer_prop1(query, template_label, templates, hints)
    threshold = extract_threshold(query)

    return {
        "nnp1": nnp1,
        "nnp2": nnp2,
        "nnp3": nnp3,
        "prop1": prop1,
        "nn1": nn1,
        "nn2": nn2,
        "threshold": threshold,
    }


def _inline_default_relationships(cypher: str, rels: str, param_names: tuple[str, ...] = ("r", "r1", "r2")) -> str:
    if not rels:
        return cypher
    rel_list = "[" + ", ".join(f"'{r}'" for r in rels.split("|")) + "]"
    for name in param_names:
        cypher = cypher.replace(
            f"($prop1 IS NULL OR type({name}) = $prop1)",
            f"type({name}) IN {rel_list}",
        )
    return cypher


def _apply_compare_fallback(expanded: dict[str, Any], slots: dict[str, Any]) -> dict[str, Any]:
    label = expanded["template_label"]
    nnp1, nnp2 = slots.get("nnp1"), slots.get("nnp2")
    if not nnp1 or not nnp2 or label not in _COMPARE_NO_SHOCK_CYPHER:
        return expanded
    expanded["cypher"] = _COMPARE_NO_SHOCK_CYPHER[label]
    expanded["parameters"] = {"nnp1": nnp1, "nnp2": nnp2}
    expanded["missing_parameters"] = []
    expanded["query_mode"] = "compare_pair"
    return expanded


def expand_formica_template(
    query: str,
    template_label: str,
    matches_df: pd.DataFrame,
    *,
    path: str | Path = DEFAULT_TEMPLATE_PATH,
) -> dict[str, Any]:
    """Build executable Cypher from classified template + triplet slots.

    1. Load template spec from formica_query_templates.json.
    2. Assign entities to nnp1/nnp2/prop1 via extract_triplet_slots().
    3. Substitute slots into the template Cypher string.
    4. If the query shape needs it, swap in a specialized Cypher mode.
    5. If required slots are still missing, fall back to single-entity neighborhood Cypher.
    """
    store = load_templates(path)
    spec = store["templates"].get(template_label)
    if not spec:
        raise KeyError(f"Unknown Formica template: {template_label}")

    slots = extract_triplet_slots(query, template_label, matches_df, path=path)
    slots = dict(slots)
    params = {k: v for k, v in slots.items() if k in spec.get("parameters", []) and v is not None}
    for key in spec.get("parameters", []):
        params.setdefault(key, slots.get(key))

    cypher = spec["cypher"]
    if "$prop1" in cypher and params.get("prop1") is None:
        cypher = _inline_default_relationships(cypher, spec.get("default_relationships", ""))

    missing = [p for p in spec.get("parameters", []) if p.startswith("nnp") and not params.get(p)]
    expanded = {
        "template_label": template_label,
        "template_id": spec["template_id"],
        "template_class": template_label,
        "description": spec["description"],
        "cypher": cypher,
        "parameters": params,
        "triplet_slots": slots,
        "missing_parameters": missing,
        "resolved_entities": matches_df[
            ["entity", "matched_node_name", "matched_id", "matched_type"]
        ].to_dict(orient="records"),
    }

    # Pick a specialized query mode when the generic triplet is a poor fit.
    mode = None
    if template_label == "F_Simple":
        nnp1_type = _row_type(matches_df, slots.get("nnp1"))
        mode = _special_simple_mode(query, slots, matches_df) or (
            "shock_beneficiaries"
            if slots.get("prop1") == "BENEFITS_FROM" and nnp1_type in _SHOCK_TYPES
            else None
        ) or (
            "commodity_transit"
            if slots.get("prop1") == "TRANSITS" and nnp1_type in {"RAW_MATERIAL", "PRODUCT"}
            else None
        )
    elif template_label.startswith("F_Quant"):
        mode = _quant_mode(query, template_label, slots)

    if mode:
        expanded["cypher"] = _SPECIAL_SIMPLE_CYPHER[mode]
        # nn1 is irrelevant to the query that triggers global ranking (it comes
        # from whatever nnp1 got picked, which is usually a stray, wrong entity for
        # these queries -- see _quant_mode) -- always rank across Company entities.
        mode_overrides = {"nn1": "COMPANY"} if mode in ("global_ranking_max", "global_ranking_min") else None
        expanded["parameters"] = _params_for_cypher(expanded["cypher"], slots, mode_overrides)
        expanded["query_mode"] = mode
        expanded["missing_parameters"] = []
        expanded["triplet_slots"] = slots
    elif template_label == "F_Simple" and missing:
        expanded = _apply_simple_single_entity(expanded, slots, matches_df)
    elif missing and template_label in {"F_CompMore", "F_CompLess", "F_CompApprox"}:
        expanded = _apply_compare_fallback(expanded, slots)
        if expanded.get("missing_parameters") and slots.get("nnp1"):
            expanded = _apply_simple_single_entity(expanded, slots, matches_df)
    elif missing and slots.get("nnp1"):
        expanded = _apply_simple_single_entity(expanded, slots, matches_df)

    if template_label in {"F_CompMore", "F_CompLess", "F_CompApprox"} and slots.get("nnp3"):
        expanded = _apply_compare_guard(expanded)

    expanded["parameters"] = _params_for_cypher(expanded["cypher"], slots, expanded.get("parameters"))
    expanded["missing_parameters"] = [
        p for p in re.findall(r"\$(\w+)", expanded["cypher"])
        if p.startswith("nnp") and not expanded["parameters"].get(p)
    ]
    return expanded


def _apply_compare_guard(expanded: dict[str, Any]) -> dict[str, Any]:
    needle = "MATCH (a:Node {id: $nnp1}), (b:Node {id: $nnp2}), (shock:Node {id: $nnp3})"
    guard = (
        "MATCH (a:Node {id: $nnp1}), (b:Node {id: $nnp2}), (shock:Node {id: $nnp3})\n"
        "WHERE a.id <> b.id AND shock.id <> a.id AND shock.id <> b.id"
    )
    if needle in expanded["cypher"] and guard not in expanded["cypher"]:
        expanded["cypher"] = expanded["cypher"].replace(needle, guard)
    return expanded


def _apply_simple_single_entity(expanded: dict[str, Any], slots: dict[str, Any], matches_df: pd.DataFrame) -> dict[str, Any]:
    """Fallback Cypher: explore the neighborhood of a single resolved entity by type."""
    nnp1 = slots.get("nnp1")
    if not nnp1:
        return expanded
    entity_type = _row_type(matches_df, nnp1)
    expanded["query_mode"] = "single_entity"
    expanded["missing_parameters"] = []
    if entity_type == "COMPANY":
        expanded["cypher"] = (
            "MATCH (company:Node {id: $nnp1})\n"
            "OPTIONAL MATCH (company)-[r:HURT_BY|BENEFITS_FROM|HEDGES|OWNS|REGULATED_BY|"
            "CONSTRAINED_BY|PRICE_LINKED_TO|NEEDS|SOURCED_FROM]->(o:Node)\n"
            "RETURN company.label AS company_name, type(r) AS relationship, "
            "o.label AS object_name, o.type AS object_type"
        )
    elif entity_type == "SECTOR":
        expanded["cypher"] = (
            "MATCH (sector:Node {id: $nnp1})-[:CONTAINS]->(company:Node {type: 'COMPANY'})\n"
            "OPTIONAL MATCH (company)-[r:HURT_BY|BENEFITS_FROM|REGULATED_BY|CONSTRAINED_BY]->(o:Node)\n"
            "RETURN sector.label AS sector_name, company.label AS company_name, "
            "type(r) AS relationship, o.label AS object_name, o.type AS object_type"
        )
    elif entity_type == "GEOGRAPHY":
        expanded["cypher"] = (
            "MATCH (geo:Node {id: $nnp1})<-[r:TRANSITS|SOURCED_FROM]-(node:Node)\n"
            "RETURN geo.label AS geography_name, node.label AS linked_name, "
            "node.type AS linked_type, type(r) AS relationship"
        )
    elif entity_type in {"RAW_MATERIAL", "PRODUCT"}:
        if slots.get("prop1") == "BENEFITS_FROM":
            expanded["cypher"] = _SPECIAL_SIMPLE_CYPHER["shock_beneficiaries"]
        else:
            expanded["cypher"] = _SPECIAL_SIMPLE_CYPHER["commodity_transit"]
    elif entity_type in {"EVENT", "MACRO_VAR"}:
        expanded["cypher"] = (
            "MATCH (shock:Node {id: $nnp1})<-[:HURT_BY|BENEFITS_FROM|CAUSES]-(node:Node)\n"
            "RETURN shock.label AS shock_name, node.label AS linked_name, node.type AS node_type"
        )
    elif entity_type == "INVESTOR":
        expanded["cypher"] = (
            "MATCH (investor:Node {id: $nnp1})-[r:INVESTS_IN]->(target:Node)\n"
            "RETURN investor.label AS investor_name, type(r) AS relationship, "
            "target.label AS target_name, target.type AS target_type"
        )
    expanded["parameters"] = _params_for_cypher(expanded["cypher"], slots, {"nnp1": nnp1})
    return expanded


_RELAXED_RETURN = (
    "s.name AS subject_name, r.name AS relationship, o.name AS object_name, "
    "[l IN labels(o) WHERE l <> 'Entity'][0] AS object_type, r.fact AS fact"
)

# Internal Graphiti bookkeeping properties -- never a fact worth surfacing to the user.
_ATTRIBUTE_LOOKUP_EXCLUDED_KEYS = [
    "uuid", "name", "name_embedding", "group_id", "summary", "created_at", "labels",
]
_ATTRIBUTE_LOOKUP_CYPHER = (
    "MATCH (s:Entity {uuid: $nnp1})\n"
    f"UNWIND [k IN keys(s) WHERE NOT k IN {_ATTRIBUTE_LOOKUP_EXCLUDED_KEYS} "
    "AND s[k] IS NOT NULL AND s[k] <> '' | k] AS attribute\n"
    "RETURN s.name AS subject_name, attribute, s[attribute] AS value\n"
    "ORDER BY attribute"
)


def _relaxed_simple_cypher(prop1: str | None) -> str:
    """Bidirectional fallback for F_Simple: Graphiti stores every fact as a generic
    :RELATES_TO relationship with the real semantic type in r.name (never a typed
    Neo4j relationship), and edge direction follows the ontology's edge_type_map
    (e.g. Person -[GovernanceRole]-> Company), which may be the reverse of what the
    query's phrasing assumes as subject/object -- so try both directions.
    """
    if prop1:
        return (
            "MATCH (s:Entity {uuid: $nnp1})-[r:RELATES_TO]->(o:Entity)\n"
            "WHERE r.name = $prop1\n"
            f"RETURN {_RELAXED_RETURN}\n"
            "UNION\n"
            "MATCH (s:Entity {uuid: $nnp1})<-[r:RELATES_TO]-(o:Entity)\n"
            "WHERE r.name = $prop1\n"
            f"RETURN {_RELAXED_RETURN}"
        )
    return (
        "MATCH (s:Entity {uuid: $nnp1})-[r:RELATES_TO]->(o:Entity)\n"
        f"RETURN {_RELAXED_RETURN}\n"
        "UNION\n"
        "MATCH (s:Entity {uuid: $nnp1})<-[r:RELATES_TO]-(o:Entity)\n"
        f"RETURN {_RELAXED_RETURN}"
    )


def _run_cypher(driver, cypher: str, params: dict[str, Any]) -> list:
    with driver.session() as s:
        return list(s.run(cypher, **params))


def execute_formica_template(driver, expanded: dict[str, Any]) -> tuple[list, str | None]:
    """Run Cypher for the expanded template, retrying with alternates when rows are empty.

    Fallback order (stops at first non-empty result):
      primary -> compare_pair -> propagation_all_paths -> propagation_open
      -> transit_broad -> beneficiary modes -> relaxed_direction -> single_entity

    Returns (rows, strategy_name). strategy_name is None when every attempt fails.
    """
    work = dict(expanded)
    slots = work.get("triplet_slots", {})

    if work.get("missing_parameters"):
        resolved = work.get("resolved_entities") or []
        if resolved:
            work = _apply_simple_single_entity(work, slots, pd.DataFrame(resolved))
        else:
            return [], None

    params = _params_for_cypher(work["cypher"], slots, work.get("parameters"))

    try:
        rows = _run_cypher(driver, work["cypher"], params)
    except Exception:
        if work["template_label"] in {"F_CompMore", "F_CompLess", "F_CompApprox"}:
            work = _apply_compare_fallback(work, slots)
            params = _params_for_cypher(work["cypher"], slots, work.get("parameters"))
            rows = _run_cypher(driver, work["cypher"], params)
        else:
            rows = []

    if rows:
        return rows, work.get("query_mode") or "primary"

    if work["template_label"] in {"F_CompMore", "F_CompLess", "F_CompApprox"} and slots.get("nnp1") and slots.get("nnp2"):
        compare = _apply_compare_fallback(work, slots)
        compare_params = _params_for_cypher(compare["cypher"], slots, compare.get("parameters"))
        rows = _run_cypher(driver, compare["cypher"], compare_params)
        if rows:
            return rows, compare.get("query_mode") or "compare_pair"

    # Multi-hop propagation retries.
    if work.get("query_mode") in {"propagation", "path_trace"} and params.get("nnp1") and params.get("nnp2"):
        alt_params = _params_for_cypher(_SPECIAL_SIMPLE_CYPHER["propagation_all_paths"], slots, params)
        rows = _run_cypher(driver, _SPECIAL_SIMPLE_CYPHER["propagation_all_paths"], alt_params)
        if rows:
            return rows, "propagation_all_paths"

    if slots.get("prop1") == "CAUSES" and params.get("nnp1"):
        prop_mode = "propagation" if params.get("nnp2") else "propagation_open"
        if prop_mode in _SPECIAL_SIMPLE_CYPHER:
            prop_params = _params_for_cypher(_SPECIAL_SIMPLE_CYPHER[prop_mode], slots, params)
            rows = _run_cypher(driver, _SPECIAL_SIMPLE_CYPHER[prop_mode], prop_params)
            if rows:
                return rows, prop_mode

    # Transit retries: drop geo filter if chokepoint-specific match is empty.
    if work.get("query_mode") in {"transit_quant", "commodity_transit"} and params.get("nnp1"):
        broad_params = {"nnp1": params["nnp1"]}
        rows = _run_cypher(
            driver,
            "MATCH (commodity:Node {id: $nnp1})-[r:TRANSITS]->(geo:Node)\n"
            "RETURN commodity.label AS commodity_name, geo.label AS geography_name, "
            "type(r) AS relationship, r.channel AS channel",
            broad_params,
        )
        if rows:
            return rows, "transit_broad"

    # Beneficiary retries: direct exposure, producers, then multi-hop paths.
    prop1 = slots.get("prop1") or params.get("prop1")
    if prop1 == "BENEFITS_FROM" and params.get("nnp1"):
        tried = {work.get("query_mode")}
        for mode in _BENEFICIARY_FALLBACKS:
            if mode in tried:
                continue
            mode_cypher = _SPECIAL_SIMPLE_CYPHER[mode]
            mode_params = _params_for_cypher(mode_cypher, slots, params)
            rows = _run_cypher(driver, mode_cypher, mode_params)
            if rows:
                return rows, mode

    # Retry with relaxed relationship direction for simple triplet queries.
    if work["template_label"] == "F_Simple" and params.get("nnp1") and prop1 != "BENEFITS_FROM":
        relaxed = _relaxed_simple_cypher(prop1)
        relaxed_params = {"nnp1": params["nnp1"]}
        if prop1:
            relaxed_params["prop1"] = prop1
        rows = _run_cypher(driver, relaxed, relaxed_params)
        if rows:
            return rows, "relaxed_direction"

    # Attribute-lookup retry: many banking facts (ticker, exchange, legal name, ...)
    # are node PROPERTIES, not edges to another entity -- something the causal-chain
    # relationship templates above can never answer. Return every non-null custom
    # attribute on the resolved entity instead.
    if work["template_label"] == "F_Simple" and params.get("nnp1"):
        rows = _run_cypher(driver, _ATTRIBUTE_LOOKUP_CYPHER, {"nnp1": params["nnp1"]})
        if rows:
            return rows, "attribute_lookup"

    # Final fallback: entity-centric neighborhood query.
    resolved = work.get("resolved_entities") or []
    if resolved and params.get("nnp1"):
        fallback = _apply_simple_single_entity(dict(work), slots, pd.DataFrame(resolved))
        fallback_params = _params_for_cypher(fallback["cypher"], slots, fallback.get("parameters"))
        rows = _run_cypher(driver, fallback["cypher"], fallback_params)
        if rows:
            return rows, "single_entity"

    return [], None
