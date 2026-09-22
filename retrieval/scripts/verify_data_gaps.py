"""Ground-truth data-gap verification: for each benchmark query, check directly
against Neo4j (the company's own neighboring nodes/edges) whether the requested
fact actually exists in the graph, rather than trusting the pipeline's own
summary text (which can be wrong in both directions -- it sometimes says "not
covered" when a broader graph check would find the fact under a slightly
different relation name or neighboring node, and it can occasionally produce a
wrong-but-confident-sounding answer that never says "not covered" at all).

Method per row:
  1. Parse resolved_entities + triplet_slots from the batch CSV.
  2. Identify the subject company's uuid and the candidate relation names
     (prop1_list) the query seems to be asking about.
  3. Query Neo4j directly: does this company have ANY edge whose relation name
     is in that candidate list (optionally further narrowed to specific target
     entities/metric-name-contains when the row identified one)? This re-checks
     existence independently of whatever narrow Cypher template the pipeline
     happened to run at retrieval time, so a code-level retrieval miss doesn't
     get mistaken for a graph-level data gap.
  4. Falls back to a phrase-based check of the pipeline's own summary only when
     no company/relation-list could be identified from the row at all (cross-
     company aggregate queries, or a totally failed entity resolution).

Usage (run from research_and_development/retrieval/scripts):
  python verify_data_gaps.py --input ../../data/full_benchmark_results_v4.csv \
      --keep-output ../../data/queries_no_data_gap.csv \
      --gap-output ../../data/queries_with_data_gap.csv \
      --report-output ../../data/data_gap_verification_report.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from formica_retrieval.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER  # noqa: E402

_GAP_PHRASES = [
    "not cover", "do not contain", "does not contain", "not disclos",
    "not available", "not found", "not provide", "not specified", "not confirmed",
    "no data", "not reported", "not mention", "not identify", "cannot be determined",
    "cannot be made", "fails to", "unable to", "not include", "not state", "not indicate",
    "insufficient", "not address", "lacks the", "lack the", "not determined", "does not detail",
    "not detail", "no information", "not clear from", "not explicitly", "genuinely unclear",
    "was not found", "were not found", "not verified", "could not be confirmed",
]
_GAP_RE = re.compile("|".join(re.escape(p) for p in _GAP_PHRASES), re.I)
_DATE_WORD_RE = re.compile(r"\b(date|dated|period)\b", re.I)


def _has_real_gap(summary: str) -> bool:
    """gap-phrase check, but a sentence solely about a missing DATE (a known,
    tolerated secondary gap on top of an otherwise-complete answer -- "branches
    and ATMs" templates, ownership-percentage templates, etc. routinely can't
    date-stamp an otherwise-fine figure) is dropped first, in either word order
    ("does not cover the date" / "the date ... was not found"), so it doesn't
    sink a row whose primary fact was actually answered.
    """
    sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
    kept = [
        s for s in sentences
        if not (_GAP_RE.search(s) and _DATE_WORD_RE.search(s))
    ]
    remaining = " ".join(kept)
    if not remaining.strip():
        # The whole summary was just "date not covered" -- nothing else was said.
        return True
    return bool(_GAP_RE.search(remaining))


def _parse_pyliteral(text):
    """resolved_entities / triplet_slots / cypher_parameters are stored as
    Python repr (single-quoted dicts), not JSON -- ast.literal_eval handles
    that; falls back to json.loads for any that happen to be valid JSON too.
    """
    if pd.isna(text) or not str(text).strip():
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None


def check_company_relation_exists(driver, company_uuid: str, prop1_list: list[str],
                                   metric_name: str | None) -> tuple[bool, int]:
    """Ground-truth existence check: does this company have ANY edge matching
    one of the candidate relation names?

    Deliberately ignores target_ids: it's built from whatever the NER/gazetteer
    pipeline resolved for the WHOLE query, which this session has repeatedly
    found includes wrong, unrelated entities (an NER fragment landing on some
    stray Shareholder/Geography node, "MD & CEO" splitting and matching noise,
    etc.) -- requiring o.uuid to be one of those polluted ids produced a false
    "no data" verdict for "Who is the current MD & CEO of State Bank of India?"
    even though the real GovernanceRole/HOLDS_ROLE fact was sitting right there
    on a different, correct object node. This mirrors why the pipeline's own
    _relaxed_simple_cypher() fallback also skips target_ids for the same reason.

    metric_name IS kept as an optional narrowing signal when present: it comes
    from a clean METRIC-type gazetteer match (not a mixed-type NER sweep), so
    it's reliable, and without it an overloaded relation name used for many
    different metrics (HAS_METRIC, MetricObservation) would make every
    metric question about a company look "covered" just because the company
    has SOME metric fact, not necessarily the one actually asked about.
    """
    if not company_uuid or not prop1_list:
        return None, 0
    with driver.session() as s:
        query = """
        MATCH (c:Entity {uuid: $company_uuid})-[r:RELATES_TO]-(o:Entity)
        WHERE r.name IN $prop1_list
          AND ($metric_name IS NULL OR toLower(o.name) CONTAINS toLower($metric_name))
        RETURN count(*) AS c
        """
        result = s.run(
            query,
            company_uuid=company_uuid,
            prop1_list=prop1_list,
            metric_name=metric_name,
        )
        count = list(result)[0]["c"]
        return count > 0, count


def classify_row(driver, row: pd.Series) -> dict:
    resolved = _parse_pyliteral(row.get("resolved_entities")) or []
    slots = _parse_pyliteral(row.get("triplet_slots")) or {}
    template_label = row.get("predicted_template")

    # F_GlobalRank / F_GroupAggregate intentionally have no single subject
    # company -- they rank/aggregate across the WHOLE company universe (or a
    # BankingSegment). Falling back to "grab any company-typed entity from
    # resolved_entities" here would pin the ground-truth check to some
    # unrelated company the entity resolver happened to also pick up (e.g. one
    # mentioned only in passing), checking whether THAT ONE company has the
    # metric -- which has nothing to do with whether the cross-company answer
    # the pipeline actually gave is correct. Skip straight to the summary
    # fallback for these two labels instead.
    company_uuid = None
    if template_label not in {"F_GlobalRank", "F_GroupAggregate"}:
        nnp1 = slots.get("nnp1")
        if nnp1:
            for e in resolved:
                if e.get("matched_id") == nnp1 and e.get("matched_type") == "COMPANY":
                    company_uuid = nnp1
                    break
        if not company_uuid:
            for e in resolved:
                if e.get("matched_type") == "COMPANY":
                    company_uuid = e.get("matched_id")
                    break

    prop1_list = slots.get("prop1_list") or []
    metric_name = slots.get("metric_name")

    exists, count = check_company_relation_exists(driver, company_uuid, prop1_list, metric_name)

    if exists is not None:
        method = "graph_check"
        is_gap = not exists
    else:
        # No company + relation-list to check directly (cross-company aggregate
        # queries, or entity resolution found nothing at all) -- fall back to
        # the pipeline's own summary text.
        method = "summary_fallback"
        summary = row.get("summary")
        kg_row_count = row.get("kg_row_count")
        if pd.isna(kg_row_count) or kg_row_count == 0:
            is_gap = True
        elif pd.isna(summary) or not str(summary).strip():
            is_gap = True
        else:
            is_gap = _has_real_gap(str(summary))

    return {
        "is_gap": is_gap,
        "method": method,
        "company_uuid": company_uuid,
        "prop1_list": prop1_list,
        "graph_match_count": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--keep-output", required=True)
    parser.add_argument("--gap-output", required=True)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    results = []
    for _, row in df.iterrows():
        results.append(classify_row(driver, row))

    result_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    keep = df[~df["is_gap"]]
    gap = df[df["is_gap"]]

    keep[["query"]].rename(columns={"query": "text"}).to_csv(args.keep_output, index=False)
    gap[["query"]].rename(columns={"query": "text"}).to_csv(args.gap_output, index=False)

    if args.report_output:
        df[["query", "is_gap", "method", "company_uuid", "prop1_list", "graph_match_count",
            "kg_row_count", "summary", "judge_label"]].to_csv(args.report_output, index=False)

    print(f"Total: {len(df)}")
    print(f"No data gap (kept): {len(keep)}")
    print(f"Data gap (excluded): {len(gap)}")
    print(df["method"].value_counts())
    print(df.groupby("method")["is_gap"].value_counts())

    driver.close()


if __name__ == "__main__":
    main()
