"""Conformance gate between ingestion and retrieval.

The graph is the interface between two independent LLM pipelines -- Graphiti's
extraction writes it, the retrieval templates read it -- and nothing has ever
enforced that contract. Every re-ingestion has silently changed it: relation
names appear under new spellings (MetricObservation vs METRIC_OBSERVATION in the
same graph), new relation names appear that no Cypher template will ever filter
on, and facts lose fields. Each time, the failure surfaced only as a drop in
benchmark score, days later, after manual digging.

This script makes those changes loud and immediate. Run it after every
ingestion; it exits non-zero when the graph drifts outside the contract.

  python3 scripts/check_graph_contract.py
  python3 scripts/check_graph_contract.py --max-unmapped-share 0.05
  python3 scripts/check_graph_contract.py --fix-casing     # normalise r.name spellings
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neo4j import GraphDatabase  # noqa: E402

from formica_retrieval.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER  # noqa: E402
from formica_retrieval.routing.template_classes import RELATION_KEYWORDS  # noqa: E402
from formica_retrieval.linking.relation_names import _screaming_snake  # noqa: E402

RELATION_COUNTS = (
    "MATCH ()-[r:RELATES_TO]-() WHERE r.name IS NOT NULL "
    "RETURN r.name AS name, count(*) AS n ORDER BY n DESC"
)
FIELD_COVERAGE = """
MATCH ()-[r:RELATES_TO]-()
RETURN count(*) AS total,
       sum(CASE WHEN r.name IS NULL THEN 1 ELSE 0 END) AS missing_name,
       sum(CASE WHEN r.fact IS NULL THEN 1 ELSE 0 END) AS missing_fact
"""
NODE_COVERAGE = """
MATCH (n:Entity)
RETURN count(*) AS total,
       sum(CASE WHEN n.name IS NULL THEN 1 ELSE 0 END) AS missing_name,
       sum(CASE WHEN size([l IN labels(n) WHERE l <> 'Entity']) = 0 THEN 1 ELSE 0 END) AS missing_type
"""
COMPANY_COUNT = "MATCH (c:Company) RETURN count(*) AS n"


def mapped_names() -> set[str]:
    """Every relation spelling the retrieval layer can currently filter on."""
    names = set(RELATION_KEYWORDS)
    for name in list(names):
        variant = _screaming_snake(name)
        if variant:
            names.add(variant)
    return names


def check(driver, max_unmapped_share: float) -> list[str]:
    problems: list[str] = []
    with driver.session() as session:
        counts = Counter({r["name"]: r["n"] for r in session.run(RELATION_COUNTS)})
        fields = session.run(FIELD_COVERAGE).single()
        nodes = session.run(NODE_COVERAGE).single()
        companies = session.run(COMPANY_COUNT).single()["n"]

    known = mapped_names()
    total_edges = sum(counts.values())
    unmapped = Counter({name: n for name, n in counts.items() if name not in known})
    unmapped_edges = sum(unmapped.values())
    share = unmapped_edges / total_edges if total_edges else 0.0

    print(f"Companies:          {companies}")
    print(f"Entity nodes:       {nodes['total']} "
          f"(missing name: {nodes['missing_name']}, missing type label: {nodes['missing_type']})")
    print(f"RELATES_TO edges:   {fields['total']} "
          f"(missing r.name: {fields['missing_name']}, missing r.fact: {fields['missing_fact']})")
    print(f"Relation names:     {len(counts)} distinct, {len(unmapped)} unmapped "
          f"({unmapped_edges} edges, {share * 100:.1f}%)")

    if unmapped:
        print("\nUnmapped relation names (invisible to every Cypher template):")
        for name, n in unmapped.most_common(25):
            print(f"  {name:38s} {n:5d}")

    # Casing drift: the same concept written two ways in one graph.
    by_canonical: dict[str, list[str]] = {}
    for name in counts:
        by_canonical.setdefault(_screaming_snake(name) or name.upper(), []).append(name)
    drifted = {k: v for k, v in by_canonical.items() if len(v) > 1}
    if drifted:
        print("\nCasing drift (same relation under multiple spellings):")
        for canonical, spellings in sorted(drifted.items()):
            detail = ", ".join(f"{s} ({counts[s]})" for s in spellings)
            print(f"  {canonical}: {detail}")
        problems.append(f"{len(drifted)} relation(s) written under multiple spellings")

    if companies == 0:
        problems.append("no :Company nodes -- retrieval cannot resolve any entity")
    if fields["missing_name"]:
        problems.append(f"{fields['missing_name']} edges have no r.name")
    if nodes["missing_type"]:
        problems.append(f"{nodes['missing_type']} entity nodes have no type label")
    if share > max_unmapped_share:
        problems.append(
            f"unmapped relations cover {share * 100:.1f}% of edges "
            f"(limit {max_unmapped_share * 100:.1f}%)"
        )
    return problems


def fix_casing(driver) -> None:
    """Rewrite SCREAMING_SNAKE r.name values to the PascalCase spelling.

    Removes the drift at the source rather than compensating for it on every
    read (which is what relation_names._with_casing_variants does today).
    """
    with driver.session() as session:
        counts = Counter({r["name"]: r["n"] for r in session.run(RELATION_COUNTS)})
        by_canonical: dict[str, list[str]] = {}
        for name in counts:
            by_canonical.setdefault(_screaming_snake(name) or name.upper(), []).append(name)
        for spellings in by_canonical.values():
            if len(spellings) < 2:
                continue
            # Prefer the PascalCase spelling -- that is what edge_types declares.
            canonical = sorted(spellings, key=lambda s: ("_" in s, -counts[s]))[0]
            for other in spellings:
                if other == canonical:
                    continue
                session.run(
                    "MATCH ()-[r:RELATES_TO]-() WHERE r.name = $old SET r.name = $new",
                    old=other, new=canonical,
                )
                print(f"  {other} -> {canonical} ({counts[other]} edges)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-unmapped-share", type=float, default=0.05)
    parser.add_argument("--fix-casing", action="store_true", help="Normalise relation-name spellings in place.")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        if args.fix_casing:
            print("Normalising relation-name casing:")
            fix_casing(driver)
            print()
        problems = check(driver, args.max_unmapped_share)
    finally:
        driver.close()

    if problems:
        print("\nCONTRACT VIOLATIONS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nGraph conforms to the retrieval contract.")


if __name__ == "__main__":
    main()
