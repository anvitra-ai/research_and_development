"""Run a template's Cypher against Neo4j, with the fallback cascade.

The primary query is only the first attempt: when it returns nothing, or returns
rows the relevance gate judges off-topic, progressively broader strategies are
tried (compare_pair, relaxed_direction, attribute_lookup) and the best-scoring
candidate is kept. Also does the Python-side aggregation that moved out of
Cypher -- ranking, segment averages, peer comparison.
"""

from __future__ import annotations

from typing import Any, Callable

from .cypher_templates import _ATTRIBUTE_LOOKUP_CYPHER, _RELAXED_RETURN
from .fact_search import _run_cypher
from .relevance import RELEVANCE_FLOOR
from .row_values import _segment_of, _usable_value
from .slot_extraction import _WANTS_GROWTH_RE, _fact_is_growth, _params_for_cypher
from .strategies import GATED as _GATED_STRATEGIES
from .template_expansion import _apply_compare_fallback, _relaxed_simple_cypher


def execute_formica_template(
    driver,
    expanded: dict[str, Any],
    scorer: Callable[[list], float] | None = None,
) -> tuple[list, str | None]:
    """Run Cypher for the expanded template, retrying with alternates when rows are empty.

    Fallback order (stops at the first ACCEPTED result):
      primary -> compare_pair -> relaxed_direction -> attribute_lookup

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

    # An unfilled slot is not fatal: the primary cypher below simply matches
    # nothing (it is run inside a try/except), and the entity-anchored retries
    # further down -- relaxed_direction, attribute_lookup -- only need nnp1.
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
            def _in_requested_segment(row: dict) -> bool:
                return _segment_of(row.get("subject_name"), row.get("graph_segment")) in segment_names

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
            val = _usable_value(row.get(value_key), row.get("fact"))
            if val is None:
                continue
            row[value_key] = val
            scored.append((val, row))
        rows = [r for _, r in sorted(scored, key=lambda pair: pair[0], reverse=reverse)][:1]

    if rows and work.get("query_mode") in (
        "group_rank_by_segment_max", "group_rank_by_segment_min", "group_vs_group",
    ):
        # Group by segment, dedupe to one value per company, then average -- see
        # the cypher comment for why this moved out of Cypher's avg(), and
        # _segment_of / _usable_value for how membership and nulls are resolved.
        by_segment: dict[str, dict[str, float]] = {}
        for row in rows:
            row = dict(row)
            company = row.get("company_name")
            if not company:
                continue
            seg = _segment_of(company, row.get("graph_segment"))
            if not seg:
                continue
            bucket = by_segment.setdefault(seg, {})
            if company in bucket:
                continue
            val = _usable_value(row.get("value"), row.get("fact"))
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
        first = dict(rows[0])
        # A comp_to_group_avg query is always about a percentage/ratio metric,
        # so an absolute amount with no "%" in the fact text is not usable.
        own_val = _usable_value(first.get("value"), first.get("fact"))
        subject_name = first.get("subject_name")
        own_segment = _segment_of(subject_name, first.get("own_graph_segment"))
        peer_vals: dict[str, float] = {}
        if own_segment:
            for row in rows:
                row = dict(row)
                peer_name = row.get("peer_name")
                if not peer_name or peer_name in peer_vals:
                    continue
                if _segment_of(peer_name, row.get("peer_graph_segment")) != own_segment:
                    continue
                pval = _usable_value(row.get("peer_value"), row.get("peer_fact"))
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

    prop1 = slots.get("prop1") or params.get("prop1")

    # Retry with relaxed relationship direction for simple triplet queries. Also
    # applies to F_LogUnion: when nnp2 is unresolved (very common -- the query's
    # "or" was actually joining two relation-name synonyms for one entity, e.g.
    # "weakness or vulnerability", not two distinct entities), F_LogUnion's own
    # cypher has nothing left to fall back on and returns zero rows outright,
    # unlike F_Simple which has this whole retry cascade.
    if work["template_label"] in {"F_Simple", "F_LogUnion"} and params.get("nnp1"):
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

    # Nothing cleared the relevance floor -- fall back to the best-scoring set
    # seen, so gating can only reorder preferences and never return fewer rows
    # than the old first-non-empty cascade.
    if fallback_best is not None:
        _, rows, strategy = fallback_best
        return rows, strategy
    return [], None
