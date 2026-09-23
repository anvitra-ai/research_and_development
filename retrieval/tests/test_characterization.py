"""Characterization tests: pin current retrieval behaviour before refactoring.

These are not specification tests -- they assert what the pipeline does TODAY, so
that a structural change (splitting template_resolver, moving slot logic, etc.)
can be proven behaviour-preserving. A failure here after a refactor means the
refactor changed something; a failure after an intentional behaviour change
means the snapshot needs regenerating (scripts/regen_snapshot.py).

Deliberately covers only the deterministic, offline part of the pipeline --
classification, slot extraction, relation inference, Cypher construction. No
Neo4j, no Gemini, no network, so it runs in seconds and in CI.

    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from formica_retrieval.search.relevance import RELEVANCE_FLOOR, merge_rows, rank_rows, row_text
from formica_retrieval.evaluation.retrieval_eval import (
    attribute_outcome,
    extract_names,
    extract_numbers,
    material_tokens,
    retrieval_recall,
)
from formica_retrieval.linking.relation_names import _screaming_snake, _with_casing_variants
from formica_retrieval.routing.template_classifier import rule_label_formica
from formica_retrieval.search.template_resolver import (
    infer_prop1_list,
)

SNAPSHOT = Path(__file__).parent / "snapshots" / "query_behaviour.json"


# --------------------------------------------------------------------------
# Snapshot: template label + inferred relations for a fixed query set.
# --------------------------------------------------------------------------

def load_snapshot() -> list[dict]:
    if not SNAPSHOT.exists():
        pytest.skip(f"snapshot missing -- run scripts/regen_snapshot.py ({SNAPSHOT})")
    return json.loads(SNAPSHOT.read_text())


@pytest.mark.parametrize("case", load_snapshot() if SNAPSHOT.exists() else [], ids=lambda c: c["id"])
def test_query_behaviour_unchanged(case):
    assert rule_label_formica(case["query"]) == case["rule_label"]
    assert sorted(infer_prop1_list(case["query"])) == sorted(case["prop1_list"])


# --------------------------------------------------------------------------
# Relation-name casing drift (the bug class that broke every Axis Bank compare).
# --------------------------------------------------------------------------

def test_screaming_snake_converts_pascal_case():
    assert _screaming_snake("MetricObservation") == "METRIC_OBSERVATION"
    assert _screaming_snake("SupportedByRelation") == "SUPPORTED_BY_RELATION"


def test_screaming_snake_leaves_non_pascal_alone():
    assert _screaming_snake("HAS_METRIC") is None
    assert _screaming_snake("CAUSES") is None


def test_casing_variants_keep_original_and_add_twin():
    out = _with_casing_variants(["MetricObservation", "HAS_METRIC"])
    assert "MetricObservation" in out
    assert "METRIC_OBSERVATION" in out
    assert "HAS_METRIC" in out


def test_casing_variants_do_not_duplicate():
    out = _with_casing_variants(["MetricObservation", "METRIC_OBSERVATION"])
    assert out.count("METRIC_OBSERVATION") == 1


# --------------------------------------------------------------------------
# retrieval_eval: the measurement axis must not drift silently either.
# --------------------------------------------------------------------------

def test_numbers_normalise_across_formatting():
    assert extract_numbers("reported 218,399 crore") == extract_numbers("reported 218399 crore")


def test_small_bare_integers_are_ignored_but_percentages_are_not():
    assert extract_numbers("3 items") == set()
    assert "3" in extract_numbers("3% growth")


def test_fiscal_year_labels_do_not_leak_digits():
    # "FY26" must not contribute a bogus 26 that matches unrelated figures.
    assert extract_numbers("in FY26") == set()


def test_sentence_initial_capital_is_not_a_name():
    assert "value" not in extract_names("Value was high.")
    assert "crisil" in extract_names("It was rated by CRISIL.")


def test_query_tokens_are_excluded_from_material_tokens():
    tokens = material_tokens("HDFC Bank's NIM was 3.46%.", "What was HDFC Bank's NIM?")
    assert "3.46" in tokens["numbers"]
    assert not any("hdfc" in n for n in tokens["names"])


def test_recall_is_none_when_nothing_material_to_check():
    assert retrieval_recall("It was not disclosed.", "some hop path", "q")["recall"] is None


def test_recall_counts_only_found_tokens():
    stats = retrieval_recall("Profit was 12.5% and 44.2%.", "profit grew 12.5 percent", "q")
    assert stats["recall"] == 0.5


@pytest.mark.parametrize(
    "label,rows,recall,expected",
    [
        ("Relevant", 5, 0.0, "answered"),
        ("Not relevant", 0, None, "no_retrieval"),
        ("Not relevant", 5, 0.0, "retrieval_miss"),
        ("Not relevant", 5, 0.5, "partial_retrieval"),
        ("Not relevant", 5, 1.0, "generation_miss"),
        ("Not relevant", 5, None, "unscored"),
    ],
)
def test_outcome_attribution(label, rows, recall, expected):
    assert attribute_outcome(label, rows, recall) == expected


# --------------------------------------------------------------------------
# relevance: row flattening and merging.
# --------------------------------------------------------------------------

def test_row_text_prefers_stored_fact():
    assert row_text({"fact": "X grew 5%", "subject_name": "X"}) == "X grew 5%"


def test_row_text_falls_back_to_triple():
    text = row_text({"subject_name": "X", "relationship": "OPERATES_IN", "object_name": "India"})
    assert "OPERATES_IN" in text and "India" in text


# One representative row per shape format_kg_rows_for_summary can emit. Keep in
# step with that function's branches: a shape it can print but row_text flattens
# to nothing is scored blank by the relevance gate, which is how `profile` and
# `avg_value` rows silently became invisible to ranking.
_SUMMARY_ROW_SHAPES = {
    "profile": {"subject_name": "UCO Bank", "profile": "UCO Bank operates 3,200 branches."},
    "ranked_max": {"subject_name": "BoB", "object_name": "return on assets", "max_value": 0.25},
    "ranked_min": {"subject_name": "BoB", "object_name": "gross NPA", "min_value": 1.1},
    "attribute": {"subject_name": "X", "attribute": "ticker", "value": "XBANK"},
    "segment_avg": {"subject_name": "X", "segment_name": "PSU", "segment_average": 3.4},
    "aggregate": {"subject_name": "Public Sector Banks", "avg_value": 3.41, "company_count": 12},
    "fact": {"subject_name": "X", "fact": "X grew 5%"},
    "fact_list": {"subject_name": "X", "facts": ["a", "b"]},
    "triple": {"subject_name": "X", "relationship": "OPERATES_IN", "object_name": "India"},
}


@pytest.mark.parametrize("shape", sorted(_SUMMARY_ROW_SHAPES))
def test_row_text_covers_summary_shapes(shape):
    """Every formattable row shape must flatten to text the scorer can read.

    Asserts more than non-emptiness: the text must survive masking of the entity
    name, since best_score() masks resolved company names before scoring and a
    row whose only content WAS the name then scores against the empty string.
    """
    row = _SUMMARY_ROW_SHAPES[shape]
    text = row_text(row)
    assert text.strip(), f"{shape} row flattened to empty text"
    masked = text.replace(str(row.get("subject_name", "")), "").strip()
    assert masked, f"{shape} row has no content beyond its subject name"


def test_merge_rows_deduplicates():
    a = [{"subject_name": "X", "fact": "f1"}]
    b = [{"subject_name": "X", "fact": "f1"}, {"subject_name": "X", "fact": "f2"}]
    assert len(merge_rows(a, b)) == 2


def test_merge_rows_respects_cap():
    a = [{"subject_name": "X", "fact": "f0"}]
    b = [{"subject_name": "X", "fact": f"f{i}"} for i in range(1, 20)]
    assert len(merge_rows(a, b, cap=5)) == 5


def test_relevance_floor_is_within_unit_range():
    assert 0.0 < RELEVANCE_FLOOR < 1.0


# --------------------------------------------------------------------------
# Embedding-dependent behaviour, run only when the model is available locally.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    st = pytest.importorskip("sentence_transformers")
    try:
        return st.SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as exc:  # offline / no cached model
        pytest.skip(f"embedding model unavailable: {exc}")


def test_rank_rows_puts_on_topic_row_first(embedder):
    rows = [
        {"fact": "The bank operates 5,000 branches across India."},
        {"fact": "Net interest margin stood at 3.46% in Q1."},
    ]
    ranked = rank_rows("What was the net interest margin?", rows, embedder)
    assert "3.46" in ranked[0]["fact"]


def test_off_topic_rows_score_below_floor(embedder):
    from formica_retrieval.search.relevance import best_score

    rows = [{"fact": "The company was incorporated in 1969 under the Companies Act."}]
    assert best_score("What is the ticker symbol?", rows, embedder) < RELEVANCE_FLOOR


# --------------------------------------------------------------------------
# Strategy registry: the names the pipeline emits must all be declared.
# --------------------------------------------------------------------------

def _emitted_strategy_names() -> set[str]:
    """Strategy names assigned anywhere in the package source.

    Deliberately a source scan rather than a curated list: a curated one is the
    very thing that drifts. Matches the three ways a strategy name is produced --
    `strategy = "..."`, `accept(rows, "...")`, and a direct write to the result
    record. The leading \b matters: without it this also matched Hugging Face's
    unrelated `aggregation_strategy="simple"` kwarg.
    """
    import re
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "src" / "formica_retrieval"
    names: set[str] = set()
    for path in pkg.rglob("*.py"):
        if path.name == "strategies.py":
            continue
        src = path.read_text()
        names |= set(re.findall(r'\bstrategy\s*=\s*"([a-z_]+)"', src))
        names |= set(re.findall(r'accept\([^,]+,\s*"([a-z_]+)"\)', src))
        names |= set(re.findall(r'\["cypher_strategy"\]\s*=\s*"([a-z_]+)"', src))
        names |= set(re.findall(r'\["query_mode"\]\s*=\s*"([a-z_]+)"', src))
    return names


def test_strategy_registry_is_complete():
    """Every strategy name the code emits is declared in the registry.

    Guards the failure this registry exists to prevent: adding a strategy and
    forgetting one of the behavioural sets keyed off its name, so it silently
    skips the relevance gate or the supplement step.
    """
    from formica_retrieval.search import strategies

    unregistered = _emitted_strategy_names() - strategies.NAMES
    assert not unregistered, (
        f"strategy names emitted but not declared in search/strategies.py: "
        f"{sorted(unregistered)}"
    )


def test_named_cypher_modes_are_all_registered():
    """Every named special mode is a strategy the pipeline can report.

    Checked against _SPECIAL_SIMPLE_CYPHER's keys directly rather than by
    scanning source: those keys ARE the definitive list of named modes, so this
    cannot drift the way a regex over call sites can.
    """
    from formica_retrieval.search import strategies
    from formica_retrieval.search.cypher_templates import _SPECIAL_SIMPLE_CYPHER

    unregistered = set(_SPECIAL_SIMPLE_CYPHER) - strategies.NAMES
    assert not unregistered, f"named Cypher modes missing from the registry: {sorted(unregistered)}"


def test_strategy_sets_are_subsets_of_the_registry():
    from formica_retrieval.search import strategies

    for label, group in (
        ("GATED", strategies.GATED),
        ("SUPPLEMENTABLE", strategies.SUPPLEMENTABLE),
        ("NODE_RETURNING", strategies.NODE_RETURNING),
    ):
        assert group <= strategies.NAMES, f"{label} names a strategy that is not registered"
    assert len(strategies.CASCADE_ORDER) == len(strategies.NAMES), "duplicate strategy name"
