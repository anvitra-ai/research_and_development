"""Result-record shape, CSV/JSONL serialization, and batch summary stats.

process_query() always returns a dict with exactly the RESULT_FIELDS keys (via
empty_query_result() as the starting point) so every consumer -- batch CSV rows,
the API response model, ad-hoc REPL use -- can rely on the same shape whether the
query succeeded, failed, or was never fully resolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULT_FIELDS = (
    "query",
    "ground_truth",
    "gold_label",
    "predicted_template",
    "template_confidence",
    "template_id",
    "template_match",
    "entities",
    "resolved_entities",
    "masked_query",
    "cypher_parameters",
    "triplet_slots",
    "cypher_strategy",
    "kg_row_count",
    "kg_hop_path",
    "summary",
    "judge_label",
    "judge_rationale",
    # Second evaluation axis (see retrieval_eval.py): did the retrieved rows
    # actually contain the ground-truth answer? Lets a failure be attributed to
    # retrieval or to generation instead of collapsing both into one judge label.
    "retrieval_recall",
    "retrieval_tokens_wanted",
    "retrieval_tokens_found",
    "outcome",
    "error",
)


def empty_query_result(query: str, gold_label: str, ground_truth: str | None = None) -> dict[str, Any]:
    return {field: None for field in RESULT_FIELDS} | {
        "query": query,
        "ground_truth": ground_truth,
        "gold_label": gold_label,
        "kg_row_count": 0,
    }


def gold_label_from_row(row: pd.Series) -> str | None:
    """The human-assigned template class for this query, or None if there isn't one.

    Returns None rather than falling back to rule_label_formica(). That fallback
    is what made "template accuracy" meaningless: the benchmark CSV has no
    formica_label column, so every gold label was produced by the same rule
    function the classifier was trained to imitate and that overrode it at
    inference -- the metric compared a function against itself and reported
    962/962. An absent label is honest; a self-generated one is not.

    To get a real number here, add a `formica_label` column with hand-assigned
    classes for a sample of queries.
    """
    if "formica_label" in row and pd.notna(row["formica_label"]):
        return str(row["formica_label"])
    return None


def serialize_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    json_fields = ("entities", "resolved_entities", "cypher_parameters", "triplet_slots")
    row = dict(result)
    for field in json_fields:
        if row.get(field) is not None:
            row[field] = json.dumps(row[field])
    return row


def append_result(
    result: dict[str, Any],
    csv_path: Path,
    jsonl_path: Path,
    *,
    write_header: bool,
) -> None:
    pd.DataFrame([serialize_result_for_csv(result)]).to_csv(
        csv_path, mode="a", header=write_header, index=False
    )
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_summary_stats(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    final = pd.read_csv(csv_path)
    acc = None
    if "template_match" in final and final["template_match"].notna().any():
        # Only meaningful when real gold labels exist (see gold_label_from_row);
        # with none, every row is None and this stays silent rather than printing
        # a number derived from the pipeline grading itself.
        matches = final["template_match"].dropna()
        acc = matches.astype(str).str.lower().eq("true").mean()
    kg_hit = None
    if "kg_row_count" in final:
        kg_hit = (pd.to_numeric(final["kg_row_count"], errors="coerce").fillna(0) > 0).mean()
    print(f"\nSaved {len(final)} results to {csv_path}")
    if acc is not None:
        print(f"Template accuracy: {acc:.3f}")
    if kg_hit is not None:
        print(f"KG row hit rate: {kg_hit:.3f}")
    if "judge_label" in final and final["judge_label"].notna().any():
        counts = final["judge_label"].value_counts()
        total = counts.sum()
        print("\nLLM-judge relevance:")
        for label, count in counts.items():
            print(f"  {label}: {count} ({count / total * 100:.0f}%)")

    if "outcome" in final and final["outcome"].notna().any():
        # The actionable view: of everything that did NOT come back Relevant,
        # how much is retrieval's fault vs the summariser's.
        failures = final[final["outcome"].ne("answered") & final["outcome"].notna()]
        print("\nFailure attribution (non-Relevant rows):")
        if failures.empty:
            print("  none")
        else:
            for outcome, count in failures["outcome"].value_counts().items():
                print(f"  {outcome}: {count} ({count / len(failures) * 100:.0f}%)")
        scored = pd.to_numeric(final.get("retrieval_recall"), errors="coerce").dropna()
        if not scored.empty:
            print(f"\nRetrieval recall (mean over {len(scored)} scorable rows): {scored.mean():.3f}")
