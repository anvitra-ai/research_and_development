"""Cypher query text used by the Formica templates.

Pure data: every parameterised query string, the shared RETURN clauses, and the
numeric-coalesce fragments that paper over Graphiti's inconsistent value fields.
Extracted from template_resolver so that module holds slot-filling and execution
logic rather than 350 lines of embedded query text, and so a Cypher change can be
reviewed without reading Python around it.

Nothing here imports from the rest of the package -- these are strings.
"""

from __future__ import annotations

# Best-effort numeric value extraction from a Graphiti edge's flattened attributes
# (property name varies by which edge_type class populated it -- ExposureRelation,
# OwnershipStake, MetricObservation, etc. -- so try every plausible one in order).
_NUMERIC_COALESCE = (
    "coalesce(r.percentage, r.exposure_percentage, r.exposure_amount, "
    "r.ownership_percentage, r.value, r.share, r.weight, 1.0)"
)


# Same fields, no trailing 1.0 default -- used where a null result is meaningful
# (global ranking: null means "no structured value", not "value is 1.0"), because
# the caller falls back to parsing a number out of r.fact itself before giving up.
_NUMERIC_COALESCE_RAW = (
    "coalesce(r.percentage, r.exposure_percentage, r.exposure_amount, "
    "r.ownership_percentage, r.value, r.share, r.weight)"
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
    # A company with NONE of the real numeric fields populated would otherwise
    # silently fall through _NUMERIC_COALESCE's trailing ",1.0" default and tie
    # with every other such company at 1.0 -- ORDER BY's tie-break is arbitrary,
    # so whichever no-data company happens to sort first could "win" the ranking
    # purely by coincidence. Require at least one real field so the ranking is
    # only ever computed over companies that actually reported the metric.
    # Same fragmented-metric-node problem as the F_GlobalRank/GroupAggregate
    # sibling modes below (e.g. IDFC FIRST Bank's real cost-to-income fact lives
    # on a node literally named "cost-to-income remains high at 70.7%", a
    # different uuid from the canonical "cost-to-income ratio" node the gazetteer
    # resolves nnp2 to) -- match by name-CONTAINS on metric_name instead of the
    # single resolved uuid, so every company's variant node is included in the
    # ranking, not just whichever one node nnp2 happened to bind to. The third
    # OR strips a trailing " ratio" specifically (never a bare "growth" strip --
    # that was tried and reverted, since a generic leftover word like "deposit"
    # over-matched unrelated metrics, e.g. "credit-deposit ratio" outranking
    # real deposit-GROWTH facts because the wrong metric's value was numerically
    # bigger). " ratio" is safe to strip because what's left is always still
    # specific ("cost-to-income", "capital adequacy", "casa", ...) -- confirmed
    # needed: IDFC FIRST Bank's cost-to-income fact lives on a node named
    # "cost-to-income remains high at 70.7%" (no "ratio" in it at all).
    # Segment filtering moved out of Cypher's EXISTS clause into Python (see
    # execute_formica_template) for the same COMPANY_SEGMENT_FALLBACK reason
    # as group_rank_by_segment_max/min -- an EXISTS-on-graph-tag check can
    # only ever see the graph's (very incomplete) BankingSegment edges, with
    # no way to fall back to the static classification table. graph_segment
    # is returned alongside so the fallback only overrides when the graph tag
    # is missing or isn't actually one of the three ownership categories.
    "global_ranking_max": (
        "MATCH (s:Entity)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE ($nn1 IS NULL OR $nn1 IN [l IN labels(s) | toUpper(l)])\n"
        "  AND (o.uuid = $nnp2 OR ($metric_name IS NOT NULL AND (\n"
        "        toLower(o.name) CONTAINS toLower($metric_name)\n"
        "        OR toLower($metric_name) CONTAINS toLower(o.name)\n"
        "        OR toLower(o.name) CONTAINS replace(toLower($metric_name), ' ratio', '')\n"
        "      )))\n"
        "OPTIONAL MATCH (s)-[:RELATES_TO]-(seg:BankingSegment)\n"
        f"WITH s, o, r, seg, {_NUMERIC_COALESCE_RAW} AS val\n"
        "RETURN s.name AS subject_name, o.name AS object_name, val AS max_value, "
        "seg.name AS graph_segment, r.fact AS fact, r.valid_at AS valid_at"
    ),
    "global_ranking_min": (
        "MATCH (s:Entity)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE ($nn1 IS NULL OR $nn1 IN [l IN labels(s) | toUpper(l)])\n"
        "  AND (o.uuid = $nnp2 OR ($metric_name IS NOT NULL AND (\n"
        "        toLower(o.name) CONTAINS toLower($metric_name)\n"
        "        OR toLower($metric_name) CONTAINS toLower(o.name)\n"
        "        OR toLower(o.name) CONTAINS replace(toLower($metric_name), ' ratio', '')\n"
        "      )))\n"
        "OPTIONAL MATCH (s)-[:RELATES_TO]-(seg:BankingSegment)\n"
        f"WITH s, o, r, seg, {_NUMERIC_COALESCE_RAW} AS val\n"
        "RETURN s.name AS subject_name, o.name AS object_name, val AS min_value, "
        "seg.name AS graph_segment, "
        "r.fact AS fact, r.valid_at AS valid_at"
    ),
    # "Which segment has the highest/lowest average <metric>?" -- per-company raw
    # rows, NOT an AVG() in Cypher. r.value is null on plenty of real
    # MetricObservation edges (same ingestion inconsistency documented on
    # global_ranking_max/min above -- e.g. Punjab National Bank's "annualized
    # return on equity of 17.33%" has r.value = None), and Cypher's avg() just
    # silently drops those rows, which is exactly backwards: it should recover
    # the number from r.fact instead of pretending the company reported nothing.
    # Left ungrouped so the Python post-processing in execute_formica_template
    # can regex-recover nulls before averaging.
    # The graph's BankingSegment coverage is very incomplete (confirmed: most
    # of the ~39 real listed banks are either untagged or tagged only with an
    # unrelated operating-model label like "universal bank" instead of a PSU/
    # Private/SFB ownership category), so this no longer requires a segment
    # edge to find candidate companies at all -- OPTIONAL MATCH the graph tag
    # and let the Python post-processing in execute_formica_template resolve
    # the real segment via COMPANY_SEGMENT_FALLBACK whenever the graph tag is
    # missing or isn't one of the three ownership categories.
    "group_rank_by_segment_max": (
        "MATCH (s:Company)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE (toLower(o.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(o.name))\n"
        "OPTIONAL MATCH (s)-[:RELATES_TO]-(seg:BankingSegment)\n"
        "RETURN s.name AS company_name, seg.name AS graph_segment, r.value AS value, r.fact AS fact"
    ),
    "group_rank_by_segment_min": (
        "MATCH (s:Company)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE (toLower(o.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(o.name))\n"
        "OPTIONAL MATCH (s)-[:RELATES_TO]-(seg:BankingSegment)\n"
        "RETURN s.name AS company_name, seg.name AS graph_segment, r.value AS value, r.fact AS fact"
    ),
    # "How do Small Finance Banks as a group compare to Private Sector Banks on
    # <metric>?" -- same per-company/regex-fallback treatment and same
    # fallback-segment resolution, for exactly the two named segments (the
    # Python side filters down to segment_names after resolving each company).
    "group_vs_group": (
        "MATCH (s:Company)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE (toLower(o.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(o.name))\n"
        "OPTIONAL MATCH (s)-[:RELATES_TO]-(seg:BankingSegment)\n"
        "RETURN s.name AS company_name, seg.name AS graph_segment, r.value AS value, r.fact AS fact"
    ),
    # "X's <metric>, and how does it compare with the average for its segment?"
    # -- the company's own value alongside the AVG(metric) of its segment-mates
    # (peers only, company itself excluded from the average it's being judged
    # against).
    # cr.value/pr.value are dropped from the WHERE filters here on purpose (see
    # global_ranking_max/min above for the same fix and why) -- requiring a
    # structured value used to return ZERO rows outright whenever the company's
    # own fact had it null (e.g. Punjab National Bank's cost-to-income ratio:
    # "...stood at 50.31%" with cr.value never populated), producing an empty
    # summary instead of an answer. Values and per-peer facts are returned raw so
    # a Python-side regex fallback on the fact text can fill in the gaps and
    # compute the average itself.
    # Peer discovery no longer traverses FROM the subject company's own graph
    # segment edge (most companies -- confirmed directly -- have none at all,
    # which made the peer-average silently empty regardless of whether OTHER
    # companies had usable data). Instead this fetches the subject's own value
    # and EVERY other company's matching value independently, each with its
    # own OPTIONAL graph segment tag, and Python resolves both sides' real
    # segment via COMPANY_SEGMENT_FALLBACK before deciding who counts as a peer.
    # The subject's own metric match moved from a mandatory MATCH to OPTIONAL --
    # confirmed directly: Axis Bank/IDFC FIRST Bank/Federal Bank/Equitas SFB
    # all have ZERO cost-to-income fact of any kind in the graph, so the old
    # mandatory MATCH failed the whole query outright (zero rows, no error,
    # cypher_strategy stayed None) even though the SEGMENT AVERAGE -- which
    # doesn't depend on the subject's own data at all -- was still fully
    # computable and worth reporting on its own.
    "comp_to_group_avg": (
        "MATCH (c:Entity {uuid: $nnp1})\n"
        "OPTIONAL MATCH (c)-[cr:RELATES_TO]-(o:Entity)\n"
        "WHERE (toLower(o.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(o.name))\n"
        "WITH c, cr\n"
        "OPTIONAL MATCH (c)-[:RELATES_TO]-(own_seg:BankingSegment)\n"
        "WITH c, cr, own_seg\n"
        "MATCH (peer:Company)-[pr:RELATES_TO]-(o2:Entity)\n"
        "WHERE peer <> c AND (toLower(o2.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(o2.name))\n"
        "OPTIONAL MATCH (peer)-[:RELATES_TO]-(peer_seg:BankingSegment)\n"
        "RETURN c.name AS subject_name, cr.value AS value, cr.fact AS fact, cr.valid_at AS valid_at, "
        "own_seg.name AS own_graph_segment, "
        "peer.name AS peer_name, pr.value AS peer_value, pr.fact AS peer_fact, peer_seg.name AS peer_graph_segment"
    ),
    # "Which banks distribute third-party insurance / are exposed to
    # microfinance / have mutual funds as shareholders ...?" -- no single
    # subject company at all, so scan every company's prop1_list-matching
    # edges and keep the ones whose object name-CONTAINS either resolved
    # topic term (name-CONTAINS, not exact uuid, for the same
    # fragmented-node reason as metric_name elsewhere: PNB's insurance fact
    # is "life and non-life insurance", Canara's is "Insurance distribution"
    # -- different nodes, same topic).
    "list_companies_by_topic": (
        "MATCH (c:Company)-[r:RELATES_TO]-(o:Entity)\n"
        "WHERE (size($prop1_list) = 0 OR r.name IN $prop1_list)\n"
        "  AND (\n"
        "        ($topic_name1 IS NOT NULL AND toLower(o.name) CONTAINS toLower($topic_name1))\n"
        "        OR ($topic_name2 IS NOT NULL AND toLower(o.name) CONTAINS toLower($topic_name2))\n"
        "      )\n"
        "RETURN c.name AS subject_name, r.name AS relationship, o.name AS object_name, "
        "r.fact AS fact, r.valid_at AS valid_at\n"
        "ORDER BY subject_name\n"
        "LIMIT 50"
    ),
}


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


_COMPARE_METRIC_CYPHER = (
    "MATCH (a:Entity {uuid: $nnp1})-[ra:RELATES_TO]-(oa:Entity)\n"
    "WHERE ra.name IN $prop1_list\n"
    "RETURN a.name AS subject_name, ra.name AS relationship, oa.name AS object_name, "
    "ra.fact AS fact, ra.valid_at AS valid_at\n"
    "UNION\n"
    "MATCH (b:Entity {uuid: $nnp2})-[rb:RELATES_TO]-(ob:Entity)\n"
    "WHERE rb.name IN $prop1_list\n"
    "RETURN b.name AS subject_name, rb.name AS relationship, ob.name AS object_name, "
    "rb.fact AS fact, rb.valid_at AS valid_at\n"
    "LIMIT 20"
)


# Same as above but narrowed to a single named metric (name-CONTAINS, same
# fragmented-metric-node handling as the F_GlobalRank family) -- used whenever
# the query names one specific metric, so the summarizer sees just the two
# relevant rows instead of up to 40 unrelated ones.
_COMPARE_METRIC_NAMED_CYPHER = (
    "MATCH (a:Entity {uuid: $nnp1})-[ra:RELATES_TO]-(oa:Entity)\n"
    "WHERE ra.name IN $prop1_list AND (toLower(oa.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(oa.name))\n"
    "RETURN a.name AS subject_name, ra.name AS relationship, oa.name AS object_name, "
    "ra.fact AS fact, ra.valid_at AS valid_at\n"
    "UNION\n"
    "MATCH (b:Entity {uuid: $nnp2})-[rb:RELATES_TO]-(ob:Entity)\n"
    "WHERE rb.name IN $prop1_list AND (toLower(ob.name) CONTAINS toLower($metric_name) OR toLower($metric_name) CONTAINS toLower(ob.name))\n"
    "RETURN b.name AS subject_name, rb.name AS relationship, ob.name AS object_name, "
    "rb.fact AS fact, rb.valid_at AS valid_at\n"
    "LIMIT 20"
)


_RELAXED_RETURN = (
    "s.name AS subject_name, r.name AS relationship, o.name AS object_name, "
    "[l IN labels(o) WHERE l <> 'Entity'][0] AS object_type, r.fact AS fact, "
    "r.valid_at AS valid_at"
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


_TEXT_SEARCH_STOPWORDS = frozenset(
    """a an and are as at be been by does do for from has have how in into is it its of on or
    that the their there these this to was were what when where which who why with your not
    any some each most more than then they them about across all also any been both but can
    could did each every other such using dataset this bank banks company companies mention
    mentioned marked explicitly specific single common every""".split()
)


_TEXT_SEARCH_CYPHER = (
    "MATCH (c:Company)-[r:RELATES_TO]-(o:Entity)\n"
    "WHERE r.fact IS NOT NULL\n"
    "WITH c, r, o, size([kw IN $keywords WHERE toLower(r.fact) CONTAINS kw]) AS score\n"
    "WHERE score >= $min_score\n"
    "RETURN c.name AS subject_name, r.name AS relationship, o.name AS object_name, "
    "r.fact AS fact, r.valid_at AS valid_at, score\n"
    "ORDER BY score DESC, subject_name\n"
    "LIMIT 40"
)
