"""Regenerate the characterization snapshot in tests/snapshots/.

Records, for a stratified sample of benchmark queries, the rule-derived template
label and the relations inferred from the keyword map -- the deterministic,
offline decisions that a refactor of template_resolver must not change.

Run this ONLY when a behaviour change is intended, and review the diff: an
unexpected change in the snapshot is exactly the regression the tests exist to
catch.

  python3 scripts/regen_snapshot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formica_retrieval.paths import DATA_DIR  # noqa: E402
from formica_retrieval.routing.template_classifier import rule_label_formica  # noqa: E402
from formica_retrieval.search.template_resolver import infer_prop1_list  # noqa: E402

QUERIES_CSV = DATA_DIR / "banking_queries/indian_banks_993_queries.csv"
OUT = ROOT / "tests" / "snapshots" / "query_behaviour.json"
PER_LABEL = 12


def main() -> None:
    df = pd.read_csv(QUERIES_CSV)
    queries = [str(t) for t in df["text"].tolist()]

    # Stratify by rule label so rare template classes are represented at all --
    # an unstratified sample would be ~63% F_Simple and would never exercise the
    # group/ranking paths where most of the special-mode logic lives.
    by_label: dict[str, list[str]] = {}
    for q in queries:
        by_label.setdefault(rule_label_formica(q), []).append(q)

    cases = []
    for label in sorted(by_label):
        for i, q in enumerate(sorted(by_label[label])[:PER_LABEL]):
            cases.append({
                "id": f"{label}-{i}",
                "query": q,
                "rule_label": label,
                "prop1_list": sorted(infer_prop1_list(q)),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases across {len(by_label)} labels to {OUT}")
    for label in sorted(by_label):
        print(f"  {label:26s} {len(by_label[label]):4d} queries -> {min(len(by_label[label]), PER_LABEL)} sampled")


if __name__ == "__main__":
    main()
