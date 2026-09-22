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
from typing import Any, Callable

import pandas as pd
from sentence_transformers import util

from .entity_resolver import enrich_for_formica_template  # noqa: F401 — re-exported for notebooks
from .cypher_templates import (
    _ATTRIBUTE_LOOKUP_CYPHER,
    _ATTRIBUTE_LOOKUP_EXCLUDED_KEYS,
    _COMPARE_METRIC_CYPHER,
    _COMPARE_METRIC_NAMED_CYPHER,
    _COMPARE_NO_SHOCK_CYPHER,
    _NUMERIC_COALESCE,
    _NUMERIC_COALESCE_RAW,
    _RELAXED_RETURN,
    _SPECIAL_SIMPLE_CYPHER,
    _TEXT_SEARCH_CYPHER,
    _TEXT_SEARCH_STOPWORDS,
)
from .fact_search import (
    _run_cypher,
    semantic_fact_search,
    semantic_relation_names,
    text_search_fallback,
)
from .paths import DATA_DIR
from .relation_names import _screaming_snake, _with_casing_variants
from .relevance import RELEVANCE_FLOOR
from .template_classes import (
    RELATION_KEYWORDS,
    SLOT_SHOCK_PRIORITY,
    SLOT_SUBJECT_PRIORITY,
    BANKING_SEGMENT_KEYWORDS,
    COMPANY_SEGMENT_FALLBACK,
)
from .template_classifier import extract_threshold

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

# Extraction is only reliable at ingestion time for SOME episodes -- the same
# metric edge type (e.g. MetricObservation on a shared "CASA ratio" node) ends up
# with r.value populated for some companies and left null for others, even though
# the number is right there in the fact text ("IDFC FIRST Bank achieved a
# sector-leading Q1 FY27 CASA ratio of 50.8%"). A global ranking that only trusts
# r.value silently drops those companies and can crown the wrong winner. Falls
# back to the first percentage/number literal in r.fact when the structured
# field is missing.
_FACT_NUMBER_RE = re.compile(r"(?<!\d)(-?\d[\d,]*\.?\d*)\s*%")


def _numeric_value_from_fact(fact: str | None) -> float | None:
    if not fact:
        return None
    match = _FACT_NUMBER_RE.search(fact)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None

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


def infer_prop1_list(query: str, hints: dict[str, Any] | None = None) -> list[str]:
    """Every RELATION_KEYWORDS match in the query, not just the first/longest.

    "What products and services does X offer, and which customer segments does it
    serve?" asks about TWO relation types (OFFERS_PRODUCT and SERVES) -- infer_prop1
    picks only the longer/first keyword match and silently drops the other. Used to
    build a prop1_list filter (F_Simple) so a multi-part question about relation
    types, not just about entities (see target_ids), gets every relevant fact
    instead of just one.

    Unlike infer_prop1, never short-circuits on a _QUERY_HINTS match: those hints
    are legacy thematic-graph relation names (some, like HURT_BY, don't even exist
    as a real r.name in the Graphiti banking graph) built for single-value prop1,
    and their trigger words are broad substrings ("regulat", "expos", "invest",
    "own") that fire on many banking queries too. Returning only the hint there
    would silently drop every real RELATION_KEYWORDS match for that query (e.g.
    "regulatory risks" contains both "regulat" -> the legacy hint, and "risk" ->
    the real EXPOSED_TO/HAS_RISK_DRIVER banking keywords) -- so the hint is added
    to the keyword matches, never used in place of them.
    """
    hints = hints or _query_hints(query)
    q = query.lower()
    matches = [rel for rel, keywords in RELATION_KEYWORDS.items() if any(kw in q for kw in keywords)]
    if hints.get("prop1") and hints["prop1"] not in matches:
        matches.append(hints["prop1"])
    return _with_casing_variants(matches)






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


_ATTRIBUTE_STYLE_RE = re.compile(
    r"\bticker\b|\bexchange(?:s)?\b|\blegal name\b|\blisted\b|\bheadquarter|"
    r"\bincorporated\b|\bfounded\b|\bfounding\b|\bstock symbol\b",
    re.I,
)


def _special_simple_mode(query: str, slots: dict[str, Any], matched: pd.DataFrame | None = None) -> str | None:
    q = query.lower()
    nnp1_type = slots.get("nn1")
    # Node-property questions (ticker, exchange, legal name, HQ, founding date, ...)
    # have no matching relation-keyword, so prop1_list comes back empty and the
    # generic F_Simple cypher's "size(prop1_list)=0 -> no filter" fallback treats
    # that as "match everything" -- returning an arbitrary 50-row dump of the
    # company's OTHER relationships that "succeeds" (non-empty) and so never lets
    # the retry cascade reach attribute_lookup, which is the only mode that can
    # actually answer these (they're node properties, not edges at all).
    if not slots.get("prop1_list_keyword") and slots.get("nnp1") and _ATTRIBUTE_STYLE_RE.search(q):
        return "attribute_lookup_direct"
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

# "Which banks/companies distribute/have/are exposed to X" -- a cross-company
# membership question with no single subject company at all (unlike
# _GLOBAL_RANKING_RE's ranking queries, this doesn't rank a metric, it lists
# every company for which some fact holds). Reused as the trigger for
# list_companies_by_topic below, and exported (see is_cross_company_query) for
# the pipeline's cross_company_search supplement -- list_companies_by_topic only
# fires when the query ALSO resolved a linkable topic entity, so this same
# pattern needs a public name for the broader case where nothing resolved.
_WHICH_BANKS_RE = re.compile(
    r"\bwhich\b(?:\s+\w+){0,3}\s+(?:bank|banks|company|companies)\b", re.I
)


def is_cross_company_query(query: str) -> bool:
    """True for "which banks/companies ..." questions with no single named subject."""
    # \w+ in _WHICH_BANKS_RE doesn't span light punctuation, so "Which
    # regulator(s) oversee every bank..." missed the pattern purely because of
    # the parenthesis in "regulator(s)" -- strip it before matching rather than
    # keep loosening the regex itself, which is already doing double duty as
    # both this check and list_companies_by_topic's trigger.
    # Delete rather than space-replace: "regulator(s)" -> "regulators" keeps it
    # one token, whereas replacing with a space would add a spurious extra
    # word ("regulator s") that pushes past the {0,3}-word gap allowance.
    return bool(_WHICH_BANKS_RE.search(re.sub(r"[(),]", "", query)))


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


# A fact describing a CHANGE rather than a level. Used to keep growth rates out
# of level rankings (and vice versa) -- see the call site in
# execute_formica_template for the concrete failure this prevents.
_GROWTH_FACT_RE = re.compile(
    r"\bgrowth\b|\bgrew\b|\brose\b|\bincreased?\b|\bdeclined?\b|\bfell\b|"
    r"\bup\s+[\d.]+\s*%|\bdown\s+[\d.]+\s*%|\byear[- ]on[- ]year\b|\byoy\b|\bY/Y\b",
    re.I,
)

# The question itself asking about growth, in which case growth facts are the
# right thing to rank and levels are the wrong thing.
_WANTS_GROWTH_RE = re.compile(r"\bgrowth\b|\bgrew\b|\bfastest\b|\bincrease\b|\byear[- ]on[- ]year\b", re.I)


def _fact_is_growth(fact: str | None) -> bool:
    return bool(fact) and bool(_GROWTH_FACT_RE.search(str(fact)))


_MIN_DIRECTION_RE = re.compile(
    r"\blowest\b|\bminimum\b|\bsmallest\b|\bweakest\b", re.I
)


def _rank_direction(query: str) -> str:
    """'min' if the query asks for the lowest/weakest value, else 'max' (default)."""
    return "min" if _MIN_DIRECTION_RE.search(query) else "max"


def resolve_segment_ids(query: str, node_df: pd.DataFrame) -> list[str]:
    """BankingSegment node uuids named in the raw query text ("Public Sector
    Bank", "PSU bank", "Small Finance Banks", ...).

    Looked up by exact node name against the live node index rather than via
    gazetteer/NER: the query's phrasing ("public sector bank", singular) never
    literally contains the stored node name ("Public Sector Banks", plural),
    so the normal substring-matching gazetteer can't find it -- this needs its
    own canonical-name -> keyword-synonym lookup (BANKING_SEGMENT_KEYWORDS).
    """
    q = query.lower()
    ids: list[str] = []
    for canonical_name, keywords in BANKING_SEGMENT_KEYWORDS.items():
        if not any(kw in q for kw in keywords):
            continue
        match = node_df[node_df["node_name"] == canonical_name]
        if not match.empty:
            ids.append(str(match.iloc[0]["id"]))
    return ids


def resolve_segment_names(query: str) -> list[str]:
    """Same BANKING_SEGMENT_KEYWORDS match as resolve_segment_ids, but returns
    the canonical segment NAME strings directly instead of graph node uuids --
    used by group_vs_group's Python-side company->segment fallback resolution
    (COMPANY_SEGMENT_FALLBACK), which has no node uuids to look up.
    """
    q = query.lower()
    return [name for name, keywords in BANKING_SEGMENT_KEYWORDS.items() if any(kw in q for kw in keywords)]


def _group_mode(query: str, template_label: str, slots: dict[str, Any]) -> str | None:
    """Cross-company / group-aggregation query shapes -- see formica_template_classes.py's
    F_GlobalRank / F_GroupAggregate / F_CompToGroupAverage docstrings for what each covers.
    """
    if template_label == "F_GlobalRank":
        if not slots.get("nnp2"):
            return None
        return "global_ranking_min" if _rank_direction(query) == "min" else "global_ranking_max"
    if template_label == "F_GroupAggregate":
        if not slots.get("nnp2"):
            return None
        # Exactly two BankingSegment matches -> compare those two directly.
        # Zero or one -> rank across ALL segments ("which segment has the
        # highest average X" doesn't name a segment at all; it's asking which
        # one wins).
        if len(slots.get("segment_ids") or []) >= 2:
            return "group_vs_group"
        return "group_rank_by_segment_min" if _rank_direction(query) == "min" else "group_rank_by_segment_max"
    if template_label == "F_CompToGroupAverage":
        if not slots.get("nnp1") or not slots.get("nnp2"):
            return None
        return "comp_to_group_avg"
    return None


def extract_triplet_slots(
    query: str,
    template_label: str,
    matches_df: pd.DataFrame,
    *,
    path: str | Path = DEFAULT_TEMPLATE_PATH,
    extra_relations: list[str] | None = None,
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
        # "Compare the net interest margin of X and Y" names ONE specific metric,
        # but _COMPARE_METRIC_CYPHER without this used to dump EVERY HAS_METRIC/
        # MetricObservation fact for both companies (up to 20 each) -- a cheap
        # summarizer model reliably lost the one relevant row (e.g. #23 of 23)
        # in that noise and reported the metric as "not covered" even though it
        # was present verbatim. Same name-CONTAINS metric resolution as the
        # F_GlobalRank family above, so the Cypher can narrow to just the two
        # matching rows instead of relying on the LLM to find them.
        metric_id = _pick_by_type(matched, {"METRIC"}, ordered=ordered)
        metric_name = None
        if metric_id:
            hit = matched[matched["matched_id"] == metric_id]
            if not hit.empty:
                metric_name = str(hit.iloc[0]["entity"])
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
    elif template_label in {"F_GlobalRank", "F_GroupAggregate", "F_CompToGroupAverage"}:
        # nnp2 = the named metric being ranked/averaged/compared (shared by all
        # three). nnp1 = the one company being checked against its segment's
        # average -- only meaningful for F_CompToGroupAverage; the other two
        # rank/aggregate across the whole company universe, so there's no
        # single "subject" company to pin down.
        metric_id = _pick_by_type(matched, {"METRIC"}, ordered=ordered)
        nnp1 = (
            _pick_by_priority(matched, ["COMPANY"], ordered=ordered)
            if template_label == "F_CompToGroupAverage"
            else None
        )
        nnp2 = metric_id
        nn1 = _row_type(matched, nnp1)
        nn2 = _row_type(matched, nnp2)
        # The same metric concept (e.g. "net interest margin") routinely ends up
        # split across several near-duplicate Metric nodes from different
        # ingestion episodes ("net interest margin" vs. "domestic net interest
        # margin" vs., in a couple of cases tonight, a whole garbled sentence)
        # -- an exact-uuid match only ever sees whichever ONE node the gazetteer
        # happened to pick, so a cross-company AVERAGE built on it silently
        # undercounts. Cypher for these three modes matches by name-contains on
        # this text instead, which pulls in the near-duplicates too.
        metric_name = None
        if metric_id:
            hit = matched[matched["matched_id"] == metric_id]
            if not hit.empty:
                metric_name = str(hit.iloc[0]["entity"])
    else:
        nnp3 = shock_id

    nnp2 = _dedupe_slots(nnp1, nnp2, matched, ordered, query)
    nn2 = _row_type(matched, nnp2)

    # Relation-keyword matching is plain substring search over the raw query,
    # so a company whose own NAME happens to contain a keyword produces a false
    # match having nothing to do with the question's actual intent -- confirmed
    # directly: "Indian Overseas Bank" always matched OPERATES_IN's "overseas"
    # keyword (branches/ATMs, ticker/exchange, and any other question about
    # that company all silently picked up an irrelevant OPERATES_IN filter).
    # Masking out resolved COMPANY name mentions before keyword matching fixes
    # this at the source, for every keyword, not just "overseas" specifically.
    keyword_query = query
    for _, crow in matched[matched["matched_type"] == "COMPANY"].iterrows():
        name_text = str(crow.get("entity") or "")
        if name_text:
            keyword_query = re.sub(re.escape(name_text), " ", keyword_query, flags=re.IGNORECASE)

    prop1 = locals().get("prop1") or infer_prop1(keyword_query, template_label, templates, hints)
    threshold = extract_threshold(query)

    # Multi-part questions ("gross NPA AND net NPA", "ROA AND ROE") resolve BOTH
    # metrics as separate entities, but nnp2 above only keeps one -- collect every
    # other resolved entity too, so F_Simple can target all of them at once
    # instead of silently answering only the first and dropping the second.
    target_ids = [
        str(mid) for mid in matched["matched_id"].tolist()
        if mid and mid != nnp1
    ]

    # Same idea, one level down: "products AND services... AND customer segments"
    # asks about two different RELATION TYPES, not two entities -- prop1 above only
    # keeps one. prop1_list carries every relation type the query seems to ask
    # about, so F_Simple can match any of them instead of just the first.
    prop1_list = infer_prop1_list(keyword_query, hints)
    # Kept separately because "did a KEYWORD match a relation?" is a different
    # question from "is prop1_list non-empty?", and several downstream decisions
    # depend on the former. _special_simple_mode's node-property check
    # (attribute_lookup_direct for ticker/exchange/legal-name questions) keys off
    # "no relation keyword matched, so this must be asking about a node property
    # rather than an edge" -- once the semantic relations below started populating
    # prop1_list unconditionally, that guard silently stopped firing and every
    # ticker query fell through to the generic F_Simple cypher instead (22 such
    # queries failed in v17 as a result).
    prop1_list_keyword = list(prop1_list)
    # Union in relation names derived from embedding search (see
    # semantic_relation_names): keyword matches stay authoritative and these only
    # widen the filter, so a relation nobody wrote a keyword for is still
    # reachable by the Cypher templates.
    for rel in extra_relations or []:
        if rel not in prop1_list:
            prop1_list.append(rel)

    # The resolved entity TEXT for nnp1/nnp2 (not just their uuids) -- used by
    # list_companies_by_topic (see _group_mode/_WHICH_BANKS_RE below) for a
    # "which banks/companies ..." cross-company query where nnp1 never names a
    # specific company at all (e.g. "which banks distribute third-party
    # insurance" resolves nnp1 to the PRODUCT node "third-party insurance", not
    # a company). An exact-uuid match on that one node only ever finds
    # whichever single company happens to share that exact node -- the same
    # fragmented-node problem as metric_name elsewhere in this file -- so this
    # mode re-matches by name-CONTAINS across every company instead.
    def _entity_text(entity_id):
        if not entity_id:
            return None
        hit = matched[matched["matched_id"] == entity_id]
        return str(hit.iloc[0]["entity"]) if not hit.empty else None

    return {
        "nnp1": nnp1,
        "nnp2": nnp2,
        "nnp3": nnp3,
        "prop1": prop1,
        "nn1": nn1,
        "nn2": nn2,
        "threshold": threshold,
        "target_ids": target_ids,
        "prop1_list": prop1_list,
        "prop1_list_keyword": prop1_list_keyword,
        "metric_name": locals().get("metric_name"),
        "topic_name1": _entity_text(nnp1),
        "topic_name2": _entity_text(nnp2),
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
    # "Compare the gross NPA ratio of X and Y" has no shock/event (nnp3) at all --
    # what's actually being compared is a NAMED METRIC between two companies, not
    # graph connectivity. The hop-distance fallback below answers a completely
    # different question ("which one is closer to some third entity") and was
    # always silently wrong for this whole query family; fetch each company's own
    # facts for the requested metric relation(s) instead, side by side, whenever
    # the query's prop1_list actually names one (HAS_METRIC/MetricObservation/...).
    prop1_list = slots.get("prop1_list") or []
    metric_name = slots.get("metric_name")
    if prop1_list:
        if metric_name:
            # A named metric occasionally gets captured under a HAS_WEAKNESS/
            # HAS_RISK_DRIVER edge instead of HAS_METRIC (e.g. Axis Bank's NIM:
            # "Axis Bank's net interest margin has fallen to a cycle low of
            # 3.46%, which is a clear weakness" -- same fact, filed as a
            # weakness rather than a metric observation). Safe to widen the
            # relation set here specifically because the query is still scoped
            # by the metric_name CONTAINS filter below, so this can't pull in
            # an unrelated weakness fact.
            named_prop1_list = list(dict.fromkeys(prop1_list + ["HAS_WEAKNESS", "HAS_RISK_DRIVER"]))
            expanded["cypher"] = _COMPARE_METRIC_NAMED_CYPHER
            expanded["parameters"] = {
                "nnp1": nnp1, "nnp2": nnp2, "prop1_list": named_prop1_list, "metric_name": metric_name,
            }
            expanded["query_mode"] = "compare_metric_named"
        else:
            expanded["cypher"] = _COMPARE_METRIC_CYPHER
            expanded["parameters"] = {"nnp1": nnp1, "nnp2": nnp2, "prop1_list": prop1_list}
            expanded["query_mode"] = "compare_metric"
        expanded["missing_parameters"] = []
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
    segment_ids: list[str] | None = None,
    extra_relations: list[str] | None = None,
) -> dict[str, Any]:
    """Build executable Cypher from classified template + triplet slots.

    1. Load template spec from formica_query_templates.json.
    2. Assign entities to nnp1/nnp2/prop1 via extract_triplet_slots().
    3. Substitute slots into the template Cypher string.
    4. If the query shape needs it, swap in a specialized Cypher mode.
    5. If required slots are still missing, fall back to single-entity neighborhood Cypher.

    segment_ids : list[str] | None
        BankingSegment node uuids the query text named ("Public Sector Banks",
        "Small Finance Banks", ...), resolved by the caller (needs the live node
        index, which this module doesn't have access to) via resolve_segment_ids().
        Only consulted for F_GlobalRank/F_GroupAggregate/F_CompToGroupAverage.
    """
    store = load_templates(path)
    spec = store["templates"].get(template_label)
    if not spec:
        raise KeyError(f"Unknown Formica template: {template_label}")

    slots = extract_triplet_slots(
        query, template_label, matches_df, path=path, extra_relations=extra_relations
    )
    slots = dict(slots)
    slots["segment_ids"] = segment_ids or []
    params = {k: v for k, v in slots.items() if k in spec.get("parameters", []) and v is not None}
    for key in spec.get("parameters", []):
        params.setdefault(key, slots.get(key))

    cypher = spec["cypher"]
    if "$prop1" in cypher and params.get("prop1") is None:
        cypher = _inline_default_relationships(cypher, spec.get("default_relationships", ""))

    missing = [p for p in spec.get("parameters", []) if p.startswith("nnp") and not params.get(p)]
    expanded = {
        # Carried so the execution stage can consult the original wording --
        # e.g. whether a ranking question asked about a level or about growth,
        # which decides which facts are eligible to be ranked.
        "query": query,
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
    elif template_label in {"F_GlobalRank", "F_GroupAggregate", "F_CompToGroupAverage"}:
        mode = _group_mode(query, template_label, slots)
    elif (
        template_label in {"F_LogUnion", "F_Simple"}
        and _row_type(matches_df, slots.get("nnp1")) != "COMPANY"
        and _WHICH_BANKS_RE.search(query)
        and (slots.get("topic_name1") or slots.get("topic_name2"))
    ):
        mode = "list_companies_by_topic"

    if mode:
        expanded["cypher"] = (
            _ATTRIBUTE_LOOKUP_CYPHER if mode == "attribute_lookup_direct" else _SPECIAL_SIMPLE_CYPHER[mode]
        )
        mode_overrides: dict[str, Any] = {}
        # nn1 is irrelevant to the query that triggers global ranking (it comes
        # from whatever nnp1 got picked, which is usually a stray, wrong entity for
        # these queries -- see _quant_mode) -- always rank across Company entities.
        if mode in ("global_ranking_max", "global_ranking_min"):
            mode_overrides["nn1"] = "COMPANY"
        # size($segment_ids) needs an actual list, never null, or the Cypher's
        # "size($segment_ids) = 0 OR ..." guard silently evaluates to NULL
        # (neither branch fires) instead of the intended "no filter" default.
        if "$segment_ids" in expanded["cypher"]:
            mode_overrides["segment_ids"] = slots.get("segment_ids") or []
        if mode in ("global_ranking_max", "global_ranking_min", "group_vs_group"):
            # Segment filtering for these three now happens in Python (see
            # execute_formica_template) via COMPANY_SEGMENT_FALLBACK, since
            # neither Cypher mode filters by segment_ids itself anymore --
            # needs the segment NAME(S), not the graph uuids.
            expanded["segment_names"] = resolve_segment_names(query)
        expanded["parameters"] = _params_for_cypher(expanded["cypher"], slots, mode_overrides or None)
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
    # A "($nnpN IS NULL OR ...)" guard means that slot is an optional narrowing
    # filter by design (e.g. F_Simple/F_LogUnion's nnp2 target-entity filter), not
    # a hard requirement -- it being absent should fall through to the broader,
    # already-working query, not divert into the missing-parameter fallback path.
    # Only an nnp* referenced WITHOUT such a guard (e.g. F_CompMore's bare
    # {uuid: $nnp2}) is genuinely required.
    optionally_guarded = set(re.findall(r"\(\$(\w+) IS NULL OR", expanded["cypher"]))
    expanded["missing_parameters"] = [
        p for p in re.findall(r"\$(\w+)", expanded["cypher"])
        if p.startswith("nnp") and p not in optionally_guarded and not expanded["parameters"].get(p)
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


def _relaxed_simple_cypher(prop1: str | None, prop1_list: list[str] | None = None) -> str:
    """Bidirectional fallback for F_Simple: Graphiti stores every fact as a generic
    :RELATES_TO relationship with the real semantic type in r.name (never a typed
    Neo4j relationship), and edge direction follows the ontology's edge_type_map
    (e.g. Person -[GovernanceRole]-> Company), which may be the reverse of what the
    query's phrasing assumes as subject/object -- so try both directions.

    Filters on the full prop1_list (every relation-name synonym the query's keywords
    matched), not just the single prop1 -- ingestion assigned the same semantic
    concept different relation-type names across episodes (e.g. CEO/chairman facts
    live under GovernanceRole for some companies and HOLDS_ROLE for others), so a
    single-name filter here silently misses the fact and falls through all the way
    to the attribute_lookup last resort, which can only return node properties and
    can never answer a relationship-based question like "who is the CEO".
    """
    names = [p for p in (prop1_list or ([prop1] if prop1 else [])) if p]
    if names:
        return (
            "MATCH (s:Entity {uuid: $nnp1})-[r:RELATES_TO]->(o:Entity)\n"
            "WHERE r.name IN $prop1_list\n"
            f"RETURN {_RELAXED_RETURN}\n"
            "UNION\n"
            "MATCH (s:Entity {uuid: $nnp1})<-[r:RELATES_TO]-(o:Entity)\n"
            "WHERE r.name IN $prop1_list\n"
            f"RETURN {_RELAXED_RETURN}"
        )
    return (
        "MATCH (s:Entity {uuid: $nnp1})-[r:RELATES_TO]->(o:Entity)\n"
        f"RETURN {_RELAXED_RETURN}\n"
        "UNION\n"
        "MATCH (s:Entity {uuid: $nnp1})<-[r:RELATES_TO]-(o:Entity)\n"
        f"RETURN {_RELAXED_RETURN}"
    )










# Strategies whose rows are gated on relevance before being accepted. The
# targeted modes (compare_pair, propagation, transit, beneficiary, and every
# named query_mode) are deliberately excluded: they only fire when the query
# clearly asked for that shape, so their rows are on-topic by construction.
#
# The four gated here are the broad ones that will return SOMETHING for almost
# any query about a resolved entity, which is exactly how an unrelated fact ends
# up blocking a later strategy that would have answered correctly --
# attribute_lookup handing back legal_name/description for a source-citation
# question, or single_entity dumping an arbitrary neighbourhood.
_GATED_STRATEGIES = frozenset({"primary", "relaxed_direction", "attribute_lookup", "single_entity"})


def execute_formica_template(
    driver,
    expanded: dict[str, Any],
    scorer: Callable[[list], float] | None = None,
) -> tuple[list, str | None]:
    """Run Cypher for the expanded template, retrying with alternates when rows are empty.

    Fallback order (stops at the first ACCEPTED result):
      primary -> compare_pair -> propagation_all_paths -> propagation_open
      -> transit_broad -> beneficiary modes -> relaxed_direction -> single_entity

    `scorer` turns this from "first non-empty wins" into "first RELEVANT wins":
    given a candidate row set it returns a 0-1 relevance score, and a broad
    strategy (see _GATED_STRATEGIES) scoring below RELEVANCE_FLOOR is set aside
    rather than accepted, letting the cascade continue to a strategy that may
    actually answer the question. Nothing is thrown away -- if no strategy
    clears the floor, the best-scoring set seen is returned, so this can only
    reorder preferences, never return fewer rows than the ungated cascade.

    Returns (rows, strategy_name). strategy_name is None when every attempt fails.
    """
    work = dict(expanded)
    slots = work.get("triplet_slots", {})

    # Best sub-floor candidate seen so far, as (score, rows, strategy).
    fallback_best: tuple[float, list, str] | None = None

    def accept(rows: list, strategy: str) -> tuple[list, str] | None:
        """Accept these rows, or set them aside and signal the cascade to continue."""
        nonlocal fallback_best
        if not rows:
            return None
        if scorer is None or strategy not in _GATED_STRATEGIES:
            return rows, strategy
        score = scorer(rows)
        if score >= RELEVANCE_FLOOR:
            return rows, strategy
        if fallback_best is None or score > fallback_best[0]:
            fallback_best = (score, rows, strategy)
        return None

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

    if rows and work.get("query_mode") in ("global_ranking_max", "global_ranking_min"):
        # global_ranking_max/min now return every candidate row unranked (see
        # _NUMERIC_COALESCE_RAW) so a null structured value can fall back to a
        # number parsed out of r.fact before being ranked, instead of silently
        # losing to companies that happened to get a structured field populated.
        value_key = "max_value" if work["query_mode"] == "global_ranking_max" else "min_value"
        reverse = work["query_mode"] == "global_ranking_max"
        segment_names = work.get("segment_names")
        if segment_names:
            # Segment-scoped ranking ("Which Public Sector Bank has the
            # strongest asset quality?") -- same COMPANY_SEGMENT_FALLBACK
            # resolution as group_rank_by_segment_max/min, since the graph's
            # own BankingSegment tag is missing or wrong for most companies
            # (e.g. Indian Overseas Bank is tagged "universal bank", not
            # "Public Sector Banks", even though it clearly is one).
            _REAL_SEGMENTS = {"Public Sector Banks", "Private Sector Banks", "Small Finance Banks"}

            def _in_requested_segment(row: dict) -> bool:
                company = row.get("subject_name")
                graph_seg = row.get("graph_segment")
                seg = graph_seg if graph_seg in _REAL_SEGMENTS else COMPANY_SEGMENT_FALLBACK.get(company)
                return seg in segment_names

            rows = [r for r in rows if _in_requested_segment(dict(r))]
        _ranking_wants_growth = bool(_WANTS_GROWTH_RE.search(expanded.get("query", "") or ""))
        scored = []
        for row in rows:
            row = dict(row)
            # A LEVEL and its GROWTH RATE are different quantities that both land
            # on loosely-matched metric nodes, and ranking them together compares
            # incompatible units. Concretely: "which bank has the lowest CASA
            # ratio?" matched Karnataka Bank's "year-on-year CASA growth of
            # 12.46%" (a change) against CSB Bank's "CASA ratio of 19%" (a level)
            # and returned Karnataka Bank, because 12.46 < 19. The metric-name
            # match is deliberately loose (it strips " ratio" so a query for
            # "CASA ratio" still finds a node named "CASA"), which is what lets
            # the growth fact in -- so filter it back out here unless the question
            # actually asked about growth.
            if not _ranking_wants_growth and _fact_is_growth(row.get("fact")):
                continue
            val = row.get(value_key)
            # r.value's UNIT varies by episode even on the same-named object node --
            # "net profit" holds the absolute profit amount in crore for some
            # companies (HDFC: 19060.0) but the question asks about the Y/Y GROWTH
            # PERCENTAGE, which only appears as text in r.fact ("up 5% year-on-
            # year"). An amount in the hundreds/thousands can never be a real
            # percentage, so treat it as untrustworthy for this ranking and prefer
            # whatever percentage is actually written in the fact.
            if val is None or abs(val) > 100:
                fact_val = _numeric_value_from_fact(row.get("fact"))
                # If the fact text has no percentage to recover either (e.g.
                # IDBI Bank's "reported total advances of ₹218,399 crore" has
                # no growth rate at all), the implausible >100 raw value must
                # NOT be kept -- it would otherwise win a percentage ranking
                # purely because a currency amount is a bigger number than any
                # real percentage, exactly backwards from what "highest growth"
                # means. Drop it instead of ranking on an impossible value.
                val = fact_val
            if val is None:
                continue
            row[value_key] = val
            scored.append((val, row))
        rows = [r for _, r in sorted(scored, key=lambda pair: pair[0], reverse=reverse)][:1]

    if rows and work.get("query_mode") in (
        "group_rank_by_segment_max", "group_rank_by_segment_min", "group_vs_group",
    ):
        # Group by segment, dedupe to one value per company (regex-recovering
        # nulls same as everywhere else above), then average -- see the cypher
        # comment for why this moved out of Cypher's avg(). Segment membership
        # itself now also has a fallback: trust the graph's own BankingSegment
        # tag only when it's actually one of the three ownership categories
        # (PSU/Private/SFB) -- "universal bank" and similar operating-model
        # tags some episodes used instead don't answer this question -- and
        # fall back to the static COMPANY_SEGMENT_FALLBACK table (stable
        # public RBI classifications) whenever the graph has no useful tag at
        # all, which is the common case for this dataset.
        _REAL_SEGMENTS = {"Public Sector Banks", "Private Sector Banks", "Small Finance Banks"}
        by_segment: dict[str, dict[str, float]] = {}
        for row in rows:
            row = dict(row)
            company = row.get("company_name")
            if not company:
                continue
            graph_seg = row.get("graph_segment")
            seg = graph_seg if graph_seg in _REAL_SEGMENTS else COMPANY_SEGMENT_FALLBACK.get(company)
            if not seg:
                continue
            bucket = by_segment.setdefault(seg, {})
            if company in bucket:
                continue
            val = row.get("value")
            if val is None or abs(val) > 100:
                # Same "don't keep an implausible >100 raw value with no
                # recoverable percentage" fix as global_ranking_max/min above.
                val = _numeric_value_from_fact(row.get("fact"))
            if val is not None:
                bucket[company] = val
        if work.get("segment_names"):
            by_segment = {seg: vals for seg, vals in by_segment.items() if seg in work["segment_names"]}
        # The field is always an AVERAGE here regardless of max/min mode (mode
        # only picks which segment wins) -- it used to be labeled "max_value"/
        # "min_value" even for a computed average, which the summarizer then
        # read literally and reported as "a maximum value of X" instead of "an
        # average of X" (confirmed on "which segment has the highest average
        # net interest margin?"). "avg_value" is what it actually is.
        aggregated = [
            {"subject_name": seg, "avg_value": round(sum(vals.values()) / len(vals), 2), "company_count": len(vals)}
            for seg, vals in by_segment.items() if vals
        ]
        # Keep every segment's average, not just the winner -- ground truth for
        # "which segment has the highest average X" routinely states ALL
        # segment averages ("PSU average 14.46% (n=5); Private average ...;
        # SFB average ..."), not just the top one, and a top-1-only hop_path
        # made that structurally impossible for the summarizer to include even
        # when it wanted to.
        if work["query_mode"] == "group_rank_by_segment_max":
            aggregated.sort(key=lambda r: r["avg_value"], reverse=True)
        elif work["query_mode"] == "group_rank_by_segment_min":
            aggregated.sort(key=lambda r: r["avg_value"])
        else:
            aggregated.sort(key=lambda r: r["subject_name"] or "")
        rows = aggregated

    if rows and work.get("query_mode") == "comp_to_group_avg":
        # comp_to_group_avg now returns one row PER PEER (raw value + fact text,
        # no AVG in Cypher -- see the cypher comment above) so nulls can be
        # regex-recovered before averaging, instead of Cypher's avg() silently
        # ignoring peers with an unpopulated pr.value or the whole query
        # returning nothing when the SUBJECT's own value was null.
        _REAL_SEGMENTS = {"Public Sector Banks", "Private Sector Banks", "Small Finance Banks"}
        first = dict(rows[0])
        own_val = first.get("value")
        # Same "don't keep an implausible >100 raw value with no recoverable
        # percentage" fix as global_ranking_max/min -- a comp_to_group_avg
        # query is always about a percentage/ratio metric, so an absolute
        # amount with no "%" anywhere in the fact text isn't a usable answer.
        if own_val is None or abs(own_val) > 100:
            own_val = _numeric_value_from_fact(first.get("fact"))
        subject_name = first.get("subject_name")
        own_graph_seg = first.get("own_graph_segment")
        own_segment = (
            own_graph_seg if own_graph_seg in _REAL_SEGMENTS
            else COMPANY_SEGMENT_FALLBACK.get(subject_name)
        )
        peer_vals: dict[str, float] = {}
        if own_segment:
            for row in rows:
                row = dict(row)
                peer_name = row.get("peer_name")
                if not peer_name or peer_name in peer_vals:
                    continue
                peer_graph_seg = row.get("peer_graph_segment")
                peer_segment = (
                    peer_graph_seg if peer_graph_seg in _REAL_SEGMENTS
                    else COMPANY_SEGMENT_FALLBACK.get(peer_name)
                )
                if peer_segment != own_segment:
                    continue
                pval = row.get("peer_value")
                if pval is None or abs(pval) > 100:
                    pval = _numeric_value_from_fact(row.get("peer_fact"))
                if pval is not None:
                    peer_vals[peer_name] = pval
        segment_average = round(sum(peer_vals.values()) / len(peer_vals), 2) if peer_vals else None
        rows = [{
            "subject_name": subject_name,
            "value": own_val,
            "fact": first.get("fact"),
            "valid_at": first.get("valid_at"),
            "segment_name": own_segment,
            "segment_average": segment_average,
            "peer_count": len(peer_vals),
        }]

    if rows:
        # Broaden a narrow target_ids-filtered F_Simple hit: a compound question
        # ("branches AND ATMs", "CRAR AND CET1 ratio", "products AND customer
        # segments") only gets every sub-item into target_ids when NER/gazetteer
        # happened to recognize ALL of them as named entities -- when it recognizes
        # only one (very common: "branch network" never literally matches the
        # query's "branches", "CET1" gets mangled by NER into an unmatched
        # fragment), target_ids silently excludes the other, equally real fact,
        # even though prop1_list (the relation-type filter) would have found it
        # fine on its own. Supplement with a target_ids-free pass on the same
        # relation types and merge in whatever new facts it turns up, so a
        # resolution gap on half the question doesn't cost the answer to the
        # other half -- capped, so this stays a small top-up, not the old
        # "target_ids does nothing" flood.
        if (
            work.get("query_mode") is None
            and work["template_label"] in {"F_Simple", "F_LogUnion"}
            and params.get("target_ids")
            and params.get("prop1_list")
            and len(rows) < 5
        ):
            seen = {tuple(sorted(dict(r).items())) for r in rows}
            # LIMIT was 15 with no ORDER BY -- for a broad multi-topic question
            # ("what credit, market, operational or regulatory risks..."), a
            # single company can easily have 15+ prop1_list-matching edges
            # across unrelated topics (loans, deposits, every metric, ...), and
            # Neo4j's arbitrary return order silently cut off the specific
            # risk-type facts the question actually asked about (confirmed:
            # State Bank of India has 4 ExposureRelation risk facts -- credit,
            # interest-rate, cyber, regulatory -- but only "regulatory" survived
            # the cap, because ~15 unrelated facts filled every other slot
            # first). Raised well above this dataset's realistic per-company
            # edge count so the cap stops being the bottleneck.
            broad_cypher = (
                "MATCH (s:Entity {uuid: $nnp1})-[r:RELATES_TO]-(o:Entity)\n"
                "WHERE ($nn1 IS NULL OR $nn1 IN [l IN labels(s) | toUpper(l)])\n"
                "  AND r.name IN $prop1_list\n"
                f"RETURN {_RELAXED_RETURN}\n"
                "LIMIT 40"
            )
            broad_params = {
                k: v for k, v in params.items() if k in {"nnp1", "nn1", "prop1_list"}
            }
            extra_rows = _run_cypher(driver, broad_cypher, broad_params)
            for extra in extra_rows:
                key = tuple(sorted(dict(extra).items()))
                if key not in seen:
                    seen.add(key)
                    rows.append(extra)
        accepted = accept(rows, work.get("query_mode") or "primary")
        if accepted:
            return accepted

    if work["template_label"] in {"F_CompMore", "F_CompLess", "F_CompApprox"} and slots.get("nnp1") and slots.get("nnp2"):
        compare = _apply_compare_fallback(work, slots)
        compare_params = _params_for_cypher(compare["cypher"], slots, compare.get("parameters"))
        rows = _run_cypher(driver, compare["cypher"], compare_params)
        accepted = accept(rows, compare.get("query_mode") or "compare_pair")
        if accepted:
            return accepted
        if compare.get("query_mode") == "compare_metric_named":
            # The named metric (e.g. gazetteer-picked "net interest margin" node)
            # didn't CONTAINS-match anything for one or both companies -- same
            # fragmented-metric-node risk as F_GlobalRank. Fall back to the
            # unnamed, broader comparison rather than silently returning nothing.
            broad_slots = dict(slots)
            broad_slots["metric_name"] = None
            broad = _apply_compare_fallback(work, broad_slots)
            broad_params = _params_for_cypher(broad["cypher"], slots, broad.get("parameters"))
            rows = _run_cypher(driver, broad["cypher"], broad_params)
            accepted = accept(rows, broad.get("query_mode") or "compare_pair")
            if accepted:
                return accepted

    # Multi-hop propagation retries.
    if work.get("query_mode") in {"propagation", "path_trace"} and params.get("nnp1") and params.get("nnp2"):
        alt_params = _params_for_cypher(_SPECIAL_SIMPLE_CYPHER["propagation_all_paths"], slots, params)
        rows = _run_cypher(driver, _SPECIAL_SIMPLE_CYPHER["propagation_all_paths"], alt_params)
        accepted = accept(rows, "propagation_all_paths")
        if accepted:
            return accepted

    if slots.get("prop1") == "CAUSES" and params.get("nnp1"):
        prop_mode = "propagation" if params.get("nnp2") else "propagation_open"
        if prop_mode in _SPECIAL_SIMPLE_CYPHER:
            prop_params = _params_for_cypher(_SPECIAL_SIMPLE_CYPHER[prop_mode], slots, params)
            rows = _run_cypher(driver, _SPECIAL_SIMPLE_CYPHER[prop_mode], prop_params)
            accepted = accept(rows, prop_mode)
            if accepted:
                return accepted

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
        accepted = accept(rows, "transit_broad")
        if accepted:
            return accepted

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
            accepted = accept(rows, mode)
            if accepted:
                return accepted

    # Retry with relaxed relationship direction for simple triplet queries. Also
    # applies to F_LogUnion: when nnp2 is unresolved (very common -- the query's
    # "or" was actually joining two relation-name synonyms for one entity, e.g.
    # "weakness or vulnerability", not two distinct entities), F_LogUnion's own
    # cypher has nothing left to fall back on and returns zero rows outright,
    # unlike F_Simple which has this whole retry cascade.
    if (
        work["template_label"] in {"F_Simple", "F_LogUnion"}
        and params.get("nnp1")
        and prop1 != "BENEFITS_FROM"
    ):
        prop1_list = slots.get("prop1_list") or params.get("prop1_list") or []
        relaxed = _relaxed_simple_cypher(prop1, prop1_list)
        relaxed_params = {"nnp1": params["nnp1"]}
        if prop1_list:
            relaxed_params["prop1_list"] = prop1_list
        elif prop1:
            relaxed_params["prop1"] = prop1
        rows = _run_cypher(driver, relaxed, relaxed_params)
        accepted = accept(rows, "relaxed_direction")
        if accepted:
            return accepted

    # Attribute-lookup retry: many banking facts (ticker, exchange, legal name, ...)
    # are node PROPERTIES, not edges to another entity -- something the causal-chain
    # relationship templates above can never answer. Return every non-null custom
    # attribute on the resolved entity instead.
    if work["template_label"] in {"F_Simple", "F_LogUnion"} and params.get("nnp1"):
        rows = _run_cypher(driver, _ATTRIBUTE_LOOKUP_CYPHER, {"nnp1": params["nnp1"]})
        accepted = accept(rows, "attribute_lookup")
        if accepted:
            return accepted

    # Final fallback: entity-centric neighborhood query.
    resolved = work.get("resolved_entities") or []
    if resolved and params.get("nnp1"):
        fallback = _apply_simple_single_entity(dict(work), slots, pd.DataFrame(resolved))
        fallback_params = _params_for_cypher(fallback["cypher"], slots, fallback.get("parameters"))
        rows = _run_cypher(driver, fallback["cypher"], fallback_params)
        accepted = accept(rows, "single_entity")
        if accepted:
            return accepted

    # Nothing cleared the relevance floor -- fall back to the best-scoring set
    # seen, so gating can only reorder preferences and never return fewer rows
    # than the old first-non-empty cascade.
    if fallback_best is not None:
        _, rows, strategy = fallback_best
        return rows, strategy
    return [], None
