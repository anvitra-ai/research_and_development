"""Attribute non-Relevant queries to retrieval vs generation.

Scores an existing batch-results JSONL with retrieval_eval (no LLM calls, no
Neo4j, no API key) and prints where the failures actually are. Run it on any
historical results file to compare runs on the retrieval axis independently of
judge noise:

  python3 scripts/analyze_failures.py ../data/full_benchmark_results_grounded_v13.jsonl
  python3 scripts/analyze_failures.py run_a.jsonl run_b.jsonl      # side by side
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formica_retrieval.evaluation.retrieval_eval import RETRIEVAL_HIT_THRESHOLD, score_row  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def report(path: Path, threshold: float, examples: int) -> None:
    rows = load_rows(path)
    for row in rows:
        row.update(score_row(row, threshold=threshold))

    total = len(rows)
    labels = Counter(r.get("judge_label") for r in rows)
    outcomes = Counter(r["outcome"] for r in rows)
    failures = [r for r in rows if r["outcome"] != "answered"]

    print(f"\n=== {path.name} ({total} rows) ===")
    print("Judge labels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count} ({count / total * 100:.1f}%)")

    print(f"\nFailure attribution ({len(failures)} non-Relevant rows):")
    for outcome, count in outcomes.most_common():
        if outcome == "answered":
            continue
        print(f"  {outcome}: {count} ({count / max(len(failures), 1) * 100:.1f}%)")

    scored = [r["retrieval_recall"] for r in rows if r["retrieval_recall"] is not None]
    if scored:
        hits = sum(1 for v in scored if v >= threshold)
        print(f"\nRetrieval recall over {len(scored)} scorable rows:")
        print(f"  mean: {sum(scored) / len(scored):.3f}")
        print(f"  >= {threshold:g}: {hits} ({hits / len(scored) * 100:.1f}%)")

    if examples:
        for bucket in ("retrieval_miss", "partial_retrieval", "generation_miss"):
            sample = [r for r in failures if r["outcome"] == bucket][:examples]
            if not sample:
                continue
            print(f"\n--- {bucket} examples ---")
            for r in sample:
                print(f"  Q: {r.get('query', '')[:110]}")
                print(f"     recall={r['retrieval_recall']:.2f} missing={r['retrieval_missing_tokens'][:6]}")
                print(f"     summary: {(r.get('summary') or '')[:150]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="One or more batch-results JSONL files.")
    parser.add_argument("--threshold", type=float, default=RETRIEVAL_HIT_THRESHOLD)
    parser.add_argument("--examples", type=int, default=0, help="Print N example rows per failure bucket.")
    args = parser.parse_args()

    for path in args.results:
        report(Path(path), args.threshold, args.examples)


if __name__ == "__main__":
    main()
