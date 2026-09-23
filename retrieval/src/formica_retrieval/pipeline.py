"""Shared Formica KG query pipeline.

End-to-end flow for each natural-language question:

  1. Entity resolution (first pass)
     Link query spans to KG nodes via gazetteer, aliases, NER + semantic search.

  2. Template routing
     Pick a Formica template class (F_Simple, F_QuantCount, ...) from the query's
     keywords. A trained SVM classifier also exists (see synthetic_queries.py) but
     measured WORSE than the rules on this benchmark and is off the routing path --
     the reasoning is recorded at the call site in process_query().

  3. Entity resolution (second pass)
     Re-link with template-aware type preferences so slots get the right node types.

  4. Entity enrichment
     Pull in additional gazetteer/alias hits needed by the predicted template.

  5. Template expansion
     Map entities to triplet slots (nnp1, nnp2, prop1, …) and build Cypher.

  6. Cypher execution
     Run against Neo4j with fallbacks (propagation, transit, beneficiaries, …).

  7. Optional summarization
     Format KG rows as hop text and summarize with Gemini.

This module holds only orchestration (process_query) and its query-level helpers.
Heavy resource loading lives in resources.py, hop-path/summarization in
summarization.py, and result-record shape/serialization in results.py -- all
re-exported here so callers can keep importing everything from `pipeline`.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config as _cfg
from .linking.entity_resolver import enrich_for_formica_template, resolve_entities
from .evaluation.llm_judge import judge_pair
from .paths import DATA_DIR
from .resources import (  # noqa: F401 — re-exported for callers
    PipelineResources,
    load_fact_index,
    load_ner_pipeline,
    load_node_index,
    load_resources,
)
from .search.relevance import best_score, merge_rows, rank_rows
from .evaluation.retrieval_eval import score_row  # noqa: F401 — re-exported for callers
from .evaluation.results import (  # noqa: F401 — re-exported for callers
    RESULT_FIELDS,
    append_result,
    empty_query_result,
    gold_label_from_row,
    print_summary_stats,
    serialize_result_for_csv,
)
from .generation.summarization import format_kg_rows_for_summary, summarize_hops  # noqa: F401 — re-exported
from .routing.template_classifier import rule_label_formica
from .search.fact_search import (
    company_profile_rows,
    cross_company_search,
    entity_profile_rows,
    per_company_top_facts,
)
from .search.strategies import NODE_RETURNING, SUPPLEMENTABLE
from .search.template_resolver import (
    execute_formica_template,
    expand_formica_template,
    is_cross_company_query,
    resolve_segment_ids,
    semantic_fact_search,
    semantic_relation_names,
    text_search_fallback,
)

TRAIN_CSV = DATA_DIR / "banking_queries/indian_banks_993_queries.csv"
LEGACY_TRAIN_CSV = DATA_DIR / "indian_banks_993_queries.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "formica_pipeline_results.csv"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "formica_pipeline_results.jsonl"


def mask_entities_with_types(text: str, entity_types: list[tuple[str, str]]) -> str:
    """Replace entity spans with <TYPE> tags for the Formica classifier."""
    masked = text
    for entity, entity_type in sorted(entity_types, key=lambda x: len(x[0]), reverse=True):
        if not entity or not entity_type:
            continue
        tag = f"<{entity_type.upper()}>"
        pattern = re.compile(re.escape(entity), flags=re.IGNORECASE)
        masked = pattern.sub(tag, masked)
    return masked


def entity_types_from_matches(matches_df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(row["entity"], row["matched_type"]) for _, row in matches_df.iterrows()]


# Which strategies may be supplemented, and which answer from node properties,
# are declared once in search/strategies.py alongside the strategies themselves
# -- see that module for why these sets are derived rather than hand-maintained.
_SUPPLEMENTABLE = SUPPLEMENTABLE

# Cap on rows handed to the summariser after merging. Matches the existing LIMIT
# 40 used by the broad Cypher passes.
_MAX_MERGED_ROWS = _cfg.MAX_MERGED_ROWS

# How many semantic hits may be added on top of the template result, and how
# similar they must be. The supplement is insurance against the template query
# missing a fact -- it is not meant to replace the result, and when it dwarfs it
# the summariser loses the answer in the noise. Measured: an unbounded merge
# took mean rows/query from 5.6 to 20.3 and, while flipping 28 of 90 failing
# queries to Relevant, pushed two previously-partial answers to "not covered"
# by burying the one row that held the answer under ~25 loosely-related facts.
_MAX_SUPPLEMENT_ROWS = _cfg.MAX_SUPPLEMENT_ROWS
_SUPPLEMENT_MIN_SIMILARITY = _cfg.SUPPLEMENT_MIN_SIMILARITY

# Enumeration questions ("which banks...") get a bigger row budget than the
# 40 used elsewhere: answering one correctly requires seeing candidate facts
# from every company, so the usual anti-dilution cap works against it. Sized
# to fit ~39 banks x 2 facts plus the similarity-ranked hits.
_MAX_ENUMERATION_ROWS = _cfg.MAX_ENUMERATION_ROWS

# Below this many retrieved rows, the company's node `summary` profile is
# appended as fallback context (see company_profile_rows). Kept low on
# purpose: the profile is ~1500 chars and would otherwise crowd out precise
# facts in results that already have them. Raised from 12 after finding UCO
# Bank's profile -- which contains its ground-truth answer verbatim -- gated
# out at 14 rows.
_PROFILE_ROW_THRESHOLD = _cfg.PROFILE_ROW_THRESHOLD

_NODE_RETURNING_STRATEGIES = NODE_RETURNING

# Analytical/SWOT-style questions ("primary competitive advantage", "key
# strength", "main weakness") ask for a synthesis, not a lookup: the graph
# rarely has one edge literally tagged "competitive advantage" -- the evidence
# is scattered across several differently-typed facts (Canara Bank's advantage
# is really "operates in Karnataka" + "serves MSME customers" + "branch density
# in the south", three separate OPERATES_IN/SERVES facts, none framed as an
# advantage). Those individual facts routinely rank below unrelated but more
# literally-worded facts in cosine similarity, so the default supplement size
# (8 rows) misses them. Widened only for this query shape -- widening it
# everywhere was already measured to dilute simple lookups (see
# _MAX_SUPPLEMENT_ROWS above).
_ANALYTICAL_RE = re.compile(
    r"\b(competitive advantage|key strength|main (?:weakness|vulnerability)|"
    r"primary (?:weakness|strength|advantage)|opportunit(?:y|ies)|"
    r"\bthreats?\b|valuation[- ]relevant|risk driver)\b",
    re.I,
)
_ANALYTICAL_SUPPLEMENT_ROWS = _cfg.ANALYTICAL_SUPPLEMENT_ROWS
_ANALYTICAL_MIN_SIMILARITY = _cfg.ANALYTICAL_MIN_SIMILARITY


def _company_names(matches_df: pd.DataFrame) -> set[str]:
    """Resolved COMPANY entity names, used to scope semantic search to the right banks."""
    if matches_df.empty or "matched_type" not in matches_df:
        return set()
    companies = matches_df[matches_df["matched_type"] == "COMPANY"]
    return {str(n) for n in companies["entity"].tolist() if n}


def _ensure_fact_index(resources: "PipelineResources"):
    if resources.fact_index is None:
        resources.fact_index = load_fact_index(resources.driver, resources.embedder)
    return resources.fact_index


def _hybrid_supplement(
    query: str,
    kg_rows: list,
    strategy: str | None,
    matches_df: pd.DataFrame,
    resources: "PipelineResources",
) -> list:
    """Merge semantic fact-search hits into a template result and rank the union.

    The template query and the embedding search fail in different, complementary
    ways: Cypher can only return facts whose r.name happens to be in the
    hand-maintained keyword map (so a relation the map doesn't know is invisible
    to it), while embedding search ignores relation names entirely and matches on
    what the fact SAYS. Running only the first is what produces this benchmark's
    largest failure bucket -- rows returned that contain part of the answer or
    none of it -- so both are run and the union is ranked by relevance.

    Scoped to the resolved companies, so this adds missing facts about the right
    banks rather than well-worded facts about the wrong ones.
    """
    if strategy is not None and strategy not in _SUPPLEMENTABLE:
        return kg_rows

    subject_names = _company_names(matches_df)
    if not subject_names:
        # No company resolved: a whole-graph search here would be unscoped and
        # is already handled by the dedicated no-entities path in process_query.
        return kg_rows

    analytical = bool(_ANALYTICAL_RE.search(query))
    extra = semantic_fact_search(
        query,
        _ensure_fact_index(resources),
        resources.embedder,
        top_k=_ANALYTICAL_SUPPLEMENT_ROWS if analytical else _MAX_SUPPLEMENT_ROWS,
        min_similarity=_ANALYTICAL_MIN_SIMILARITY if analytical else _SUPPLEMENT_MIN_SIMILARITY,
        subject_names=subject_names,
    )
    if not extra:
        return kg_rows
    merged = merge_rows(kg_rows, extra, cap=_MAX_MERGED_ROWS)
    return rank_rows(
        query, merged, resources.embedder, limit=_MAX_MERGED_ROWS, mask_terms=subject_names
    )


def _finish(
    result: dict,
    query: str,
    resources: "PipelineResources",
    judge: bool,
    ground_truth: str | None = None,
    judge_votes: int = 1,
) -> dict:
    """Apply the optional LLM-as-judge step and the retrieval-recall scoring,
    then return the result.

    Every process_query() return funnels through here, so a query that fails
    entity resolution or Cypher execution (empty summary) still gets auto-labeled
    the same way llm_judge.py's standalone batch script does, instead of only
    judging the success path.
    """
    if judge:
        summary = (result.get("summary") or "").strip()
        if not summary:
            result["judge_label"] = "Not relevant"
            result["judge_rationale"] = "No summary was produced for this query (empty assistant output)."
        else:
            verdict = judge_pair(
                resources.gemini, query, summary, ground_truth=ground_truth, votes=judge_votes
            )
            result["judge_label"] = verdict.label
            result["judge_rationale"] = verdict.rationale

    # Deterministic, LLM-free second axis -- always computed (it costs nothing
    # and needs no API key), so even an unjudged run can be inspected for
    # whether retrieval is returning the answer-bearing facts at all.
    result.update(
        {k: v for k, v in score_row(result).items() if k in RESULT_FIELDS}
    )
    return result


def process_query(
    query: str,
    gold_label: str,
    resources: "PipelineResources",
    *,
    summarize: bool = False,
    judge: bool = False,
    ground_truth: str | None = None,
    judge_votes: int = 1,
    template_override: str | None = None,
) -> dict:
    """Run the full pipeline for a single query and return a result dict.

    `template_override` forces the template class instead of classifying, so a
    router can be A/B'd against another with every other stage held constant.
    """
    result = empty_query_result(query, gold_label, ground_truth)

    try:
        # --- Step 1: first-pass entity linking (no template context yet) ---
        _, matches_df = resolve_entities(
            query,
            resources.ner,
            resources.embedder,
            resources.node_df,
            resources.node_emb,
            aliases=resources.aliases,
        )
        if matches_df.empty:
            # Nothing linked at all -- typically a cross-company question that
            # names no company and no gazetteer-resolvable topic. Rather than
            # bail out with an empty summary, search every company's fact text:
            # by embedding similarity first (semantic_fact_search), falling back
            # to the cruder keyword scan only if nothing clears the floor.
            result["error"] = "no_entities_resolved"
            fallback_rows = semantic_fact_search(
                query, _ensure_fact_index(resources), resources.embedder
            )
            strategy = "semantic_fact_search"
            if not fallback_rows:
                # Nothing cleared the similarity floor -- fall back to the older
                # keyword scan rather than returning nothing at all.
                fallback_rows = text_search_fallback(resources.driver, query)
                strategy = "text_search_fallback"
            if fallback_rows:
                expanded_stub = {
                    "template_id": "text_search",
                    "template_label": "F_Simple",
                    "description": "Search across all companies' facts (no entity resolved).",
                }
                result["cypher_strategy"] = strategy
                result["kg_row_count"] = len(fallback_rows)
                kg_hop_path = format_kg_rows_for_summary(fallback_rows, expanded_stub)
                result["kg_hop_path"] = kg_hop_path or None
                if summarize and kg_hop_path and resources.gemini is not None:
                    result["summary"] = summarize_hops(resources.gemini, kg_hop_path, query)
            return _finish(result, query, resources, judge, ground_truth, judge_votes)

        masked_query = mask_entities_with_types(query, entity_types_from_matches(matches_df))

        # --- Step 2: route the query to a Formica template ---
        # Rules, not the SVM, and this is a measured choice rather than a default.
        #
        # The SVM was retrained on balanced synthetic data whose labels are true by
        # construction (synthetic_queries.py), fixing both the rare-class shortage
        # (15 of 24 classes had zero real examples) and the tautology in the old
        # setup, where it learned rule_label_formica's own output. On held-out
        # PHRASINGS it reached 0.922. It still lost the head-to-head badly:
        #
        #   mean retrieval recall, 150 queries, every other stage identical
        #     rule router        0.608        classifier router  0.533
        #     better on 0 queries, worse on 24; on the 58 disagreements 0.651 vs 0.462
        #
        # The reason is a genre gap, not a training defect. This benchmark asks
        # compound disclosure questions ("Roughly how many branches and ATMs does
        # X operate, and as of what date?"), a shape absent from CSQA and from
        # every synthetic frame, so the model lands on whatever is nearest in
        # bigram space -- that example routes to F_LogIntersection on the bare
        # " and ", and "...promoter or largest shareholder, and as of what date?"
        # routes to F_CompToGroupAverage. The rules encode real observations about
        # these specific shapes and win because of it.
        #
        # So the classifier is trained and available, but off the routing path.
        # Putting it back needs hand-labelled gold for a sample of REAL queries --
        # see gold_label_from_row() -- not more synthetic data.
        template_label = template_override or rule_label_formica(query)
        confidence = 1.0
        # --- Step 3: second-pass linking with template-aware type preferences ---
        _, matches_df = resolve_entities(
            query,
            resources.ner,
            resources.embedder,
            resources.node_df,
            resources.node_emb,
            aliases=resources.aliases,
            template_label=template_label,
        )
        result["entities"] = matches_df["entity"].tolist() if not matches_df.empty else None
        result["masked_query"] = (
            mask_entities_with_types(query, entity_types_from_matches(matches_df))
            if not matches_df.empty
            else masked_query
        )
        result["predicted_template"] = template_label
        result["template_confidence"] = float(confidence)
        # None when the dataset carries no hand-assigned gold class, so an
        # unmeasurable quantity is reported as unmeasurable rather than as 100%.
        result["template_match"] = (template_label == gold_label) if gold_label else None

        if matches_df.empty:
            result["error"] = "no_entities_resolved"
            return _finish(result, query, resources, judge, ground_truth, judge_votes)

        # --- Step 4: add missing entities the template expects (gazetteer/alias sweep) ---
        matches_df = enrich_for_formica_template(
            query, template_label, matches_df, resources.node_df, aliases=resources.aliases
        )
        result["resolved_entities"] = matches_df.to_dict(orient="records")

        # --- Step 5: fill triplet slots and build parameterized Cypher ---
        segment_ids = (
            resolve_segment_ids(query, resources.node_df)
            if template_label in {"F_GlobalRank", "F_GroupAggregate", "F_CompToGroupAverage"}
            else None
        )
        # Relation names the query semantically touches, unioned into prop1_list
        # so a relation absent from the hand-maintained keyword map is still
        # reachable by the template Cypher (24 of the graph's 72 relation names
        # are currently unmapped).
        extra_relations = semantic_relation_names(
            query,
            _ensure_fact_index(resources),
            resources.embedder,
            subject_names=_company_names(matches_df) or None,
        )
        expanded = expand_formica_template(
            query,
            template_label,
            matches_df,
            segment_ids=segment_ids,
            extra_relations=extra_relations,
        )
        result["template_id"] = expanded["template_id"]
        result["cypher_parameters"] = expanded.get("parameters")
        result["triplet_slots"] = expanded.get("triplet_slots")

        # --- Step 6: execute Cypher (retrying, and rejecting off-topic results) ---
        # Mask the resolved entity names out of the relevance score: every row
        # for this query repeats them, and unmasked they dominate the similarity
        # so completely that the gate accepts any fact about the right company.
        mask_terms = _company_names(matches_df)
        kg_rows, strategy = execute_formica_template(
            resources.driver,
            expanded,
            scorer=lambda rows: best_score(query, rows, resources.embedder, mask_terms),
        )
        # A template slot the query never filled (F_CompMore's $nnp2/$nnp3 when
        # only one entity resolved) means the TEMPLATE cannot answer -- it does
        # not mean the GRAPH cannot. Record it and keep going: steps 6b-6d below
        # are entity-anchored rather than slot-anchored and routinely answer
        # these. Returning here instead skipped them, and the only reason that
        # was survivable was an accident: a fallback rewrote the cypher to a
        # dead single-slot :Node form, which made the missing_parameters
        # recomputation come back empty and silently re-enabled the supplements.
        # Both that fallback and the :Node schema are gone; this is the fix.
        if not kg_rows and expanded.get("missing_parameters"):
            result["error"] = f"missing_parameters:{expanded['missing_parameters']}"

        # --- Step 6b: supplement template rows with semantic fact search ---
        kg_rows = _hybrid_supplement(query, kg_rows, strategy, matches_df, resources)

        # --- Step 6c: cross-company enumeration supplement ---
        # "Which banks have LIC as a shareholder?" resolves LIC (or Kerala, or
        # Fitch, or nothing at all) but no COMPANY -- so the normal cascade
        # above treats whatever DID resolve as if it were the query's single
        # subject, which is the wrong shape for an enumerate-every-matching-
        # company question. list_companies_by_topic (inside
        # execute_formica_template) only fires when a linkable topic entity
        # resolved; this catches the broader set, including queries with no
        # resolvable entity at all ("standalone vs consolidated", "rating-
        # outlook divergence"), and unlike semantic_fact_search's flat top-25
        # ranking, diversifies across companies so the answer set isn't
        # silently truncated to whichever few banks happen to phrase the fact
        # most similarly to the query.
        # Two trigger conditions, not one: the phrasing check (is_cross_company_
        # query, "which banks...") catches most cases cheaply, but "Across all 39
        # listed Indian banks in this dataset, which report the highest deposit
        # growth?" puts "which" nowhere near "bank" and slips past every regex
        # variant tried. Rather than keep chasing phrasings, also fire this
        # whenever the ENTIRE cascade above came back empty and no company
        # resolved -- at that point there is nothing to lose (kg_rows is empty
        # regardless) and a real chance the query was cross-company all along.
        cross_company_trigger = is_cross_company_query(query) or not kg_rows
        if cross_company_trigger and not mask_terms:
            fact_index = _ensure_fact_index(resources)
            extra = cross_company_search(query, fact_index, resources.embedder)
            # For an explicit "which banks..." enumeration, also guarantee every
            # company is represented (see per_company_top_facts). Similarity
            # ranking alone silently drops banks whose qualifying fact is worded
            # unusually -- and a dropped bank cannot be recovered downstream, so
            # the answer is truncated before the summariser ever sees it.
            if is_cross_company_query(query):
                extra = merge_rows(
                    extra,
                    per_company_top_facts(query, fact_index, resources.embedder),
                    cap=_MAX_ENUMERATION_ROWS,
                )
            if extra:
                had_rows_already = bool(kg_rows)
                # Enumeration needs breadth, so it gets a larger row budget than
                # a normal lookup -- the usual 40-row cap exists to stop dilution
                # of single-fact answers, which is the opposite problem here.
                cap = _MAX_ENUMERATION_ROWS if is_cross_company_query(query) else _MAX_MERGED_ROWS
                merged = merge_rows(kg_rows, extra, cap=cap)
                kg_rows = rank_rows(query, merged, resources.embedder, limit=cap)
                strategy = strategy if had_rows_already else "cross_company_search"

        # --- Step 6d: append the company's node profile as fallback context ---
        # Only when the retrieved facts are thin. The profile is dense and long,
        # so adding it to an already-rich result is the dilution pattern measured
        # earlier this session; adding it to a sparse one supplies figures that
        # were never extracted as their own edges.
        if mask_terms and len(kg_rows) < _PROFILE_ROW_THRESHOLD:
            kg_rows = kg_rows + company_profile_rows(resources.driver, mask_terms)
        elif strategy in _NODE_RETURNING_STRATEGIES:
            # These answer by describing a NODE rather than traversing edges, so
            # the node's own summary is the most on-point evidence available --
            # and it is exactly what _ATTRIBUTE_LOOKUP_EXCLUDED_KEYS strips out.
            # Covers any entity type, not just Company: "Who is <person>?" hits
            # this path and has almost no scalar properties to answer from.
            kg_rows = kg_rows + entity_profile_rows(
                resources.driver, (result.get("cypher_parameters") or {}).get("nnp1")
            )

        result["cypher_strategy"] = strategy
        result["kg_row_count"] = len(kg_rows)
        kg_hop_path = format_kg_rows_for_summary(kg_rows, expanded)
        result["kg_hop_path"] = kg_hop_path or None

        # --- Step 7 (optional): natural-language summary of KG paths ---
        if summarize and kg_hop_path and resources.gemini is not None:
            result["summary"] = summarize_hops(resources.gemini, kg_hop_path, query)

    except Exception as exc:
        result["error"] = str(exc)

    return _finish(result, query, resources, judge, ground_truth, judge_votes)
