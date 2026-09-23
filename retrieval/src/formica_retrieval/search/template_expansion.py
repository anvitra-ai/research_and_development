"""Turn triplet slots into a runnable Cypher query.

Picks the template's own Cypher or one of the named special modes, inlines
relationship types, binds parameters, and records which required slots the query
could not fill. Pure string/dict work -- nothing here touches Neo4j.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .cypher_templates import (
    _ATTRIBUTE_LOOKUP_CYPHER,
    _COMPARE_METRIC_CYPHER,
    _COMPARE_METRIC_NAMED_CYPHER,
    _COMPARE_NO_SHOCK_CYPHER,
    _RELAXED_RETURN,
    _SPECIAL_SIMPLE_CYPHER,
)
from .slot_extraction import (
    DEFAULT_TEMPLATE_PATH,
    _WHICH_BANKS_RE,
    _group_mode,
    _params_for_cypher,
    _quant_mode,
    _row_type,
    _special_simple_mode,
    extract_triplet_slots,
    load_templates,
    resolve_segment_names,
)

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
        mode = _special_simple_mode(query, slots)
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
    elif missing and template_label in {"F_CompMore", "F_CompLess", "F_CompApprox"}:
        expanded = _apply_compare_fallback(expanded, slots)

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
