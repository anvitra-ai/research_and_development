"""The retrieval strategy registry: every way a query can be answered, declared once.

WHY THIS FILE EXISTS
--------------------
A strategy's NAME is load-bearing in four separate places -- the cascade that
produces it, the relevance gate that decides whether to trust it, the supplement
step that decides whether to top it up, and the profile step that decides whether
a node summary is primary evidence. Those used to be three hand-maintained
frozensets in two modules plus fourteen bare string literals, so adding or
removing a strategy meant finding every one of them. Removing `single_entity`
missed one on the first pass. Declare a Strategy here and every set derives from
it; `test_strategy_registry_is_complete` fails if an emitted name is unregistered.

This registry covers the CASCADE strategies only -- the ones that name a way of
finding rows. The other fallback mechanisms in this codebase are deliberately not
here, because they are not interchangeable with these and share nothing but the
word "fallback". They live next to the code they protect:

  entity linking         linking/entity_resolver.py   gazetteer -> alias -> NER
                                                      -> semantic, each filling
                                                      only spans the previous
                                                      left unoccupied
  value recovery         search/row_values.py         _usable_value: regex a
                                                      number out of r.fact when
                                                      r.value is null/implausible
  segment classification search/row_values.py         _segment_of: static RBI
                                                      table when the graph's own
                                                      BankingSegment tag is
                                                      missing or operating-model
  metric-node matching   search/cypher_templates.py   name-contains instead of
                                                      uuid, because one concept
                                                      splits across near-duplicate
                                                      nodes per episode
  row supplements        pipeline.py steps 6b-6d      semantic / cross-company /
                                                      node-profile layers
  LLM transient failure  generation/summarization.py  bounded retries + timeouts
                         evaluation/llm_judge.py      (a hung connection once
                                                      wedged a whole run)
  startup + config       resources.py, config.py      NER model falls back to the
                                                      default; a malformed .env
                                                      value warns and uses the
                                                      in-code default

Tuning knobs for all of the above (floors, caps, timeouts, retry counts) live in
config.py, not here -- this file is about identity and ordering, not magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    """One way of producing rows, and how the rest of the pipeline should treat it."""

    name: str
    description: str
    # Must clear RELEVANCE_FLOOR to be accepted, instead of being taken as soon
    # as it is non-empty. Set for the BROAD strategies -- the ones that return
    # SOMETHING for almost any query about a resolved entity, which is exactly
    # how an unrelated fact ends up blocking a later strategy that would have
    # answered correctly (attribute_lookup handing back legal_name/description
    # for a source-citation question).
    gated: bool = False
    # May have semantic fact-search hits merged in. False for the synthesised
    # modes: their meaning comes from the aggregation, and appending loose facts
    # actively misleads the summariser about what was computed.
    supplementable: bool = False
    # Answers by describing a NODE rather than traversing edges, so the node's
    # own summary is the most on-point evidence available rather than filler.
    node_returning: bool = False


# Declaration order is the cascade order: execute_formica_template tries these in
# sequence and stops at the first ACCEPTED result (non-empty, and clearing the
# relevance floor if gated). Reordering this tuple does NOT reorder the cascade
# -- the control flow in template_execution still does that -- but the two must
# agree, and CASCADE_ORDER below is what a reader should trust.
STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "primary",
        "The template's own Cypher, with every slot filled.",
        gated=True,
        supplementable=True,
    ),
    Strategy(
        "compare_pair",
        "F_CompMore/Less/Approx two-entity comparison, shock term dropped.",
    ),
    Strategy(
        "compare_metric",
        "Two-entity comparison over any shared metric.",
    ),
    Strategy(
        "compare_metric_named",
        "Comparison narrowed to a named metric, matched by name-contains. Falls "
        "back to the unnamed compare_metric when the name matches nothing for "
        "one of the two companies.",
    ),
    Strategy(
        "relaxed_direction",
        "F_Simple/F_LogUnion retried ignoring edge direction: Graphiti stores "
        "facts as generic :RELATES_TO whose direction follows the ontology's "
        "edge_type_map, often the reverse of what the phrasing assumes.",
        gated=True,
        supplementable=True,
    ),
    Strategy(
        "attribute_lookup",
        "The resolved node's own scalar properties -- ticker, exchange, legal "
        "name -- which no relationship template can ever reach.",
        gated=True,
        supplementable=True,
        node_returning=True,
    ),
    Strategy(
        "attribute_lookup_direct",
        "Attribute lookup chosen up front from the question's shape, rather "
        "than reached as a fallback.",
        node_returning=True,
    ),
    # --- Named special modes: shapes the generic triplet Cypher cannot express.
    # None are gated (they are precise by construction, not broad) and none are
    # supplementable (they return synthesised rows).
    Strategy("global_ranking_max", "Highest value of one metric across all companies."),
    Strategy("global_ranking_min", "Lowest value of one metric across all companies."),
    Strategy("group_rank_by_segment_max", "Segment with the highest average for a metric."),
    Strategy("group_rank_by_segment_min", "Segment with the lowest average for a metric."),
    Strategy("group_vs_group", "Two named segments' averages, side by side."),
    Strategy("comp_to_group_avg", "One company against its own segment's peer average."),
    Strategy("list_companies_by_topic", "Every company linked to a resolved topic entity."),
    # --- Whole-query bypasses: used when the cascade has nothing to stand on.
    Strategy(
        "semantic_fact_search",
        "Embedding search over every company's fact text. Used when entity "
        "linking resolved nothing at all.",
    ),
    Strategy(
        "text_search_fallback",
        "Keyword scan over fact text: the last resort, when not even semantic "
        "search cleared its similarity floor.",
    ),
    Strategy(
        "cross_company_search",
        "Enumeration across companies, diversified so every bank is "
        "represented rather than only those phrasing a fact most similarly.",
    ),
)

_BY_NAME = {s.name: s for s in STRATEGIES}

#: Every registered strategy name. Used to validate what the pipeline emits.
NAMES = frozenset(_BY_NAME)

#: Cascade order, for documentation and for tests that pin it.
CASCADE_ORDER: tuple[str, ...] = tuple(s.name for s in STRATEGIES)

#: Must clear RELEVANCE_FLOOR rather than being accepted when merely non-empty.
GATED = frozenset(s.name for s in STRATEGIES if s.gated)

#: May have semantic fact-search hits merged in.
SUPPLEMENTABLE = frozenset(s.name for s in STRATEGIES if s.supplementable)

#: Answer from node properties, so the node `summary` is primary evidence.
NODE_RETURNING = frozenset(s.name for s in STRATEGIES if s.node_returning)


def get(name: str) -> Strategy | None:
    """The registered Strategy, or None if `name` is not one."""
    return _BY_NAME.get(name)
