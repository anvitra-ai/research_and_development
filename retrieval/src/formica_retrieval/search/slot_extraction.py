"""Choose which resolved entities fill a template's triplet slots.

Given a query, its predicted Formica template class and the entities linked to KG
nodes, decide which node becomes nnp1/nnp2, which relationship name goes in
prop1, and which specialized query mode (if any) the question really wants.
Produces the slot dict that template_expansion turns into Cypher.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import DATA_DIR
from ..linking.relation_names import _with_casing_variants
from ..routing.template_classes import (
    RELATION_KEYWORDS,
    SLOT_SUBJECT_PRIORITY,
    BANKING_SEGMENT_KEYWORDS,
)
from ..routing.template_classifier import extract_threshold

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


def _special_simple_mode(query: str, slots: dict[str, Any]) -> str | None:
    """Node-property questions are the only F_Simple shape needing a special mode.

    This used to also route propagation / path_trace / beneficiary / transit /
    sector questions, but every one of those targeted the pre-Graphiti :Node
    schema and could never return a row against the banking graph.
    """
    q = query.lower()
    # Node-property questions (ticker, exchange, legal name, HQ, founding date, ...)
    # have no matching relation-keyword, so prop1_list comes back empty and the
    # generic F_Simple cypher's "size(prop1_list)=0 -> no filter" fallback treats
    # that as "match everything" -- returning an arbitrary 50-row dump of the
    # company's OTHER relationships that "succeeds" (non-empty) and so never lets
    # the retry cascade reach attribute_lookup, which is the only mode that can
    # actually answer these (they're node properties, not edges at all).
    if not slots.get("prop1_list_keyword") and slots.get("nnp1") and _ATTRIBUTE_STYLE_RE.search(q):
        return "attribute_lookup_direct"
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

    nnp1, nn1 = _pick_subject(query, template_label, matched, hints)
    nnp2 = None
    nn2 = None

    from_id, to_id = _parse_from_to(query, matched)

    if template_label in {"F_CompMore", "F_CompLess", "F_CompApprox"}:
        nnp1 = company_ids[0] if company_ids else nnp1
        nnp2 = company_ids[1] if len(company_ids) > 1 else _pick_distinct(
            matched, {"COMPANY", "SECTOR"}, nnp1, ordered=ordered
        )
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
        if template_label == "F_QuantCount" and commodity_id:
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
    elif template_label == "F_LogUnion":
        nnp1, nn1 = _pick_subject(query, template_label, matched, hints)
        nnp2 = _pick_distinct(matched, set(matched["matched_type"]), nnp1, ordered=ordered)
        nn2 = _row_type(matched, nnp2)
    elif template_label == "F_Simple":
        if from_id and to_id:
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
