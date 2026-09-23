"""Did ingestion actually capture the source file's numbers?

check_graph_contract.py verifies the SHAPE of the graph (relation names, required
fields). This checks its CONTENT against the document it was built from, which is
a different failure and was invisible until measured: 25% of the source's numeric
facts never reached the graph, growth percentages fared worst at 62% captured,
and Punjab National Bank landed just 4 of its 31 numbers -- 14 facts in total,
every one of them qualitative, while the source carried deposits, advances, NPA
and return figures for it.

Nothing failed loudly when that happened. All 41 episodes loaded with full
content; the extractor simply declined to emit edges for most of the numbers, and
the only symptom was benchmark questions being answered "not covered". This turns
that into an exit code.

  python3 scripts/check_extraction_coverage.py
  python3 scripts/check_extraction_coverage.py --min-capture 0.85 --min-per-bank 0.5
  python3 scripts/check_extraction_coverage.py --show 15
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neo4j import GraphDatabase  # noqa: E402

from formica_retrieval.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER  # noqa: E402
from formica_retrieval.evaluation.retrieval_eval import extract_numbers  # noqa: E402

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "graphiti/data/ontology_data/banking_ontology_data.txt"
)

# Section headings look like:  STATE BANK OF INDIA (ticker SBIN, NSE and BSE)
_SECTION_RE = re.compile(r"^([A-Z][A-Z&.' ]{3,60})\s*\(ticker\s+([A-Z0-9&]+)", re.M)

# "…up 9.73% year-on-year" / "deposits 11.63%" -- the figures the benchmark's
# growth questions are made of, and the category the extractor loses most often.
_GROWTH_RE = re.compile(
    r"(deposits?|advances|loan book)[^.]{0,80}?up\s+([\d.]+)\s*%|(deposits?|advances)\s+([\d.]+)\s*%",
    re.I,
)

_COMPANY_FACTS = (
    "MATCH (c:Company)-[r:RELATES_TO]-() WHERE r.fact IS NOT NULL "
    "RETURN c.name AS company, collect(r.fact) AS facts"
)


def _tokens(name: str) -> list[str]:
    name = name.lower().replace("limited", "").replace("ltd", "").replace("the ", "")
    return re.sub(r"[^a-z ]", " ", name).split()


def match_company(source_name: str, graph_names: list[str]) -> str | None:
    """Map a source section heading to a graph Company node.

    Exact normalised match first, then the best token-containment match. Plain
    substring matching is wrong and actively misleading here: "Bank of India" is
    a substring of "State Bank of India" and will silently steal its facts,
    which makes SBI look 96% unextracted when it is fine.
    """
    want = _tokens(source_name)
    for name in graph_names:
        if _tokens(name) == want:
            return name
    best, best_score = None, 0.0
    for name in graph_names:
        have = _tokens(name)
        overlap = len(set(want) & set(have))
        if overlap and overlap == min(len(want), len(have)):
            score = overlap - abs(len(want) - len(have)) * 0.5
            if score > best_score:
                best, best_score = name, score
    return best


def parse_sections(text: str) -> dict[str, str]:
    heads = list(_SECTION_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[m.group(1).strip().title()] = text[m.start() : end]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--min-capture", type=float, default=0.85,
                        help="Fail below this overall numeric capture rate.")
    parser.add_argument("--min-growth-capture", type=float, default=0.85,
                        help="Fail below this capture rate for growth percentages.")
    parser.add_argument("--min-per-bank", type=float, default=0.40,
                        help="Fail if any bank falls below this capture rate.")
    parser.add_argument("--show", type=int, default=8, help="How many worst banks to list.")
    args = parser.parse_args()

    sections = parse_sections(Path(args.source).read_text())
    if not sections:
        print(f"No bank sections parsed from {args.source} -- has the format changed?")
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            rows = [dict(r) for r in session.run(_COMPANY_FACTS)]
    finally:
        driver.close()
    graph_numbers = {
        r["company"]: extract_numbers(" ".join(str(f) for f in r["facts"])) for r in rows
    }
    graph_names = list(graph_numbers)

    total_src = total_found = growth_src = growth_found = 0
    per_bank: list[tuple[str, int, int]] = []
    unmatched: list[str] = []

    for bank, body in sections.items():
        company = match_company(bank, graph_names)
        if company is None:
            unmatched.append(bank)
            continue
        src_nums = extract_numbers(body)
        found = src_nums & graph_numbers[company]
        total_src += len(src_nums)
        total_found += len(found)
        per_bank.append((bank, len(src_nums), len(found)))
        for m in _GROWTH_RE.finditer(body):
            value = m.group(2) or m.group(4)
            if value:
                growth_src += 1
                if f"{float(value):g}" in graph_numbers[company]:
                    growth_found += 1

    capture = total_found / total_src if total_src else 0.0
    growth_capture = growth_found / growth_src if growth_src else 1.0
    print(f"Source sections:      {len(sections)}  (matched to graph: {len(per_bank)})")
    print(f"All numeric facts:    {total_found}/{total_src} captured ({capture * 100:.1f}%)")
    print(f"Growth percentages:   {growth_found}/{growth_src} captured ({growth_capture * 100:.1f}%)")

    per_bank.sort(key=lambda t: t[2] / max(t[1], 1))
    if args.show:
        print("\nWorst-captured banks:")
        for bank, src, found in per_bank[: args.show]:
            print(f"  {bank:36s} {found:3d}/{src:3d}  ({found / max(src, 1) * 100:4.0f}%)")

    problems: list[str] = []
    if unmatched:
        problems.append(f"{len(unmatched)} source section(s) matched no Company node: {unmatched[:5]}")
    if capture < args.min_capture:
        problems.append(f"numeric capture {capture * 100:.1f}% below {args.min_capture * 100:.0f}%")
    if growth_capture < args.min_growth_capture:
        problems.append(
            f"growth-percentage capture {growth_capture * 100:.1f}% below "
            f"{args.min_growth_capture * 100:.0f}%"
        )
    starved = [(b, f, s) for b, s, f in per_bank if f / max(s, 1) < args.min_per_bank]
    if starved:
        problems.append(
            f"{len(starved)} bank(s) below {args.min_per_bank * 100:.0f}% capture: "
            + ", ".join(f"{b} ({f}/{s})" for b, f, s in starved[:5])
        )

    if problems:
        print("\nEXTRACTION COVERAGE FAILURES:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nExtraction coverage within thresholds.")


if __name__ == "__main__":
    main()
