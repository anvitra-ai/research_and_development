"""Separate retrieval recall from generation quality.

The LLM judge returns a single label per query, which conflates two very
different failures:

  * we never fetched the facts that answer the question (retrieval problem), and
  * we fetched them and the summary still didn't say them (generation problem).

Both show up as "Not relevant" and are indistinguishable in the judge output, so
there is no way to tell which half of the pipeline to invest in. This module adds
a cheap, deterministic, LLM-free second axis: does the material content of the
ground-truth answer actually APPEAR in the retrieved hop path?

Method: take the tokens that carry the answer -- numbers and proper names -- from
the ground truth, drop the ones that also appear in the question (those are given,
not answers: "What is HDFC Bank's NIM?" hands you "HDFC Bank" for free, and
counting it would make every row look well-retrieved), and check what fraction of
the remainder is present in the hop path.

This is deliberately a LOWER BOUND on true retrieval recall -- unit paraphrases
("1.2 lakh crore" vs "120,000 crore"), spelled-out numbers, and synonymous names
all count as misses. It is precise in the other direction, which is what matters
for attribution: a HIGH score is strong evidence the facts were retrieved, so a
non-Relevant judgement on a high-recall row is a generation failure, not a
retrieval one.
"""

from __future__ import annotations

import re
from typing import Any

from .. import config as _cfg

# Number with optional thousands separators and optional decimal part, e.g.
# "6.9", "218,399", "1,04,500" (Indian grouping), "2024". The leading lookbehind
# rejects digits glued to letters so fiscal-period labels ("FY26", "Q1FY26")
# don't contribute a bogus "26" that then matches any unrelated figure.
_NUMBER_RE = re.compile(r"(?<![A-Za-z\d])\d[\d,]*(?:\.\d+)?")

# Runs of capitalised words: "State Bank of India", "CRISIL", "Basel III".
# Lowercase joiners are allowed mid-run so multi-word institution names survive.
_NAME_RE = re.compile(r"\b[A-Z][\w&.'-]*(?:\s+(?:of|and|for|the|de|van)\s+[A-Z][\w&.'-]*|\s+[A-Z][\w&.'-]*)*")

# Capitalised tokens that carry no identity on their own -- almost always a
# sentence-initial function word rather than part of a name.
_NAME_STOPWORDS = {
    "the", "this", "that", "these", "those", "a", "an", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "with", "and", "or", "but", "if",
    "it", "its", "was", "were", "is", "are", "be", "been", "has", "have", "had",
    "no", "not", "none", "there", "their", "they", "he", "she", "we", "you",
    "however", "although", "while", "based", "according", "per", "both", "each",
    "however", "overall", "further", "also", "such", "during", "after", "before",
    "yes", "data", "information", "source", "note", "reference", "answer",
}


def _normalize_number(raw: str) -> str | None:
    """Canonicalise a number string so "218,399" and "218399.0" compare equal."""
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # Render without trailing zeros so 6.90 and 6.9 are the same token.
    return f"{value:g}"


def extract_numbers(text: str) -> set[str]:
    """Material numbers in `text`, normalised for comparison.

    Bare integers under 10 are dropped unless written as a percentage: they
    collide with list counters, quarter numbers and stray digits far too often,
    and a false match here would wrongly credit retrieval with content it never
    returned.
    """
    if not text:
        return set()
    out: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        normalized = _normalize_number(raw)
        if normalized is None:
            continue
        is_percent = text[match.end() : match.end() + 1] == "%"
        if "." not in normalized and abs(float(normalized)) < 10 and not is_percent:
            continue
        out.add(normalized)
    return out


def extract_names(text: str) -> set[str]:
    """Material proper names in `text`, lower-cased for comparison.

    A single capitalised word at the start of a sentence is discarded: English
    capitalises there regardless of whether the word is a name, so "Value was
    42.5%" would otherwise contribute a phantom token "value" that can never be
    found in the hop path and would drag recall down on every such row.
    Multi-word runs and mid-sentence capitals are kept -- those are capitalised
    because they ARE names.
    """
    if not text:
        return set()
    out: set[str] = set()
    for match in _NAME_RE.finditer(text):
        name = match.group(0).strip().rstrip(".,;:")
        if len(name) < 3 or name.lower() in _NAME_STOPWORDS:
            continue
        if " " not in name and _is_sentence_initial(text, match.start()) and not name.isupper():
            continue
        out.add(name.lower())
    return out


def _is_sentence_initial(text: str, start: int) -> bool:
    """True when position `start` begins the text or follows sentence-ending punctuation."""
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?:;"


def material_tokens(ground_truth: str, query: str) -> dict[str, set[str]]:
    """Answer-bearing tokens: what the ground truth says that the question didn't.

    Subtracting the query's own tokens is the whole point -- the entity the user
    named is always echoed in the hop path, so counting it would inflate recall
    on every single row.
    """
    query_numbers = extract_numbers(query)
    query_names = extract_names(query)
    numbers = extract_numbers(ground_truth) - query_numbers
    names = {
        name
        for name in extract_names(ground_truth) - query_names
        # Drop names that are a substring of something already given in the
        # question ("HDFC" when the query said "HDFC Bank").
        if not any(name in q or q in name for q in query_names)
    }
    return {"numbers": numbers, "names": names}


def retrieval_recall(ground_truth: str | None, kg_hop_path: str | None, query: str = "") -> dict[str, Any]:
    """Fraction of the ground truth's answer-bearing tokens present in the hop path.

    Returns `recall=None` when the ground truth contains no material tokens to
    look for (e.g. a purely qualitative reference answer, or one that just says
    the figure was not disclosed) -- those rows are unscorable on this axis and
    must not be averaged in as zeros.
    """
    tokens = material_tokens(ground_truth or "", query or "")
    wanted = tokens["numbers"] | tokens["names"]
    if not wanted:
        return {
            "recall": None,
            "wanted": 0,
            "found": 0,
            "missing": [],
            "numbers_wanted": 0,
            "numbers_found": 0,
        }

    hop = kg_hop_path or ""
    hop_numbers = extract_numbers(hop)
    hop_lower = hop.lower()

    found_numbers = {n for n in tokens["numbers"] if n in hop_numbers}
    found_names = {n for n in tokens["names"] if n in hop_lower}
    found = found_numbers | found_names

    return {
        "recall": len(found) / len(wanted),
        "wanted": len(wanted),
        "found": len(found),
        "missing": sorted(wanted - found),
        "numbers_wanted": len(tokens["numbers"]),
        "numbers_found": len(found_numbers),
    }


# Blaming the summariser requires that essentially ALL of the answer was sitting
# in the retrieved rows. A laxer bar (half the tokens) badly over-attributes to
# generation: the two-part questions in this benchmark ("advances AND deposit
# growth") score 0.5-0.7 when retrieval found one half and the other half is
# genuinely absent from the graph -- a data/retrieval gap that the summariser
# correctly reported as not covered.
RETRIEVAL_HIT_THRESHOLD = _cfg.RETRIEVAL_HIT_THRESHOLD

OUTCOMES = (
    "answered",
    "no_retrieval",
    "retrieval_miss",
    "partial_retrieval",
    "generation_miss",
    "unscored",
)


def attribute_outcome(
    judge_label: str | None,
    kg_row_count: int | None,
    recall: float | None,
    *,
    threshold: float = RETRIEVAL_HIT_THRESHOLD,
) -> str:
    """Assign a failed query to the stage that actually failed.

    - answered:           the judge called it Relevant; nothing to attribute.
    - no_retrieval:       Cypher and every fallback returned zero rows (reach).
    - retrieval_miss:     rows came back containing NONE of the answer -- we
                          fetched the wrong facts (precision, not reach).
    - partial_retrieval:  some of the answer came back and some didn't, i.e. a
                          multi-part question where the graph or the query
                          covered only one part.
    - generation_miss:    the whole answer was in the retrieved rows and the
                          summary still failed to convey it.
    - unscored:           no ground-truth tokens to check against.
    """
    if judge_label == "Relevant":
        return "answered"
    if not kg_row_count:
        return "no_retrieval"
    if recall is None:
        return "unscored"
    if recall >= threshold:
        return "generation_miss"
    return "retrieval_miss" if recall == 0 else "partial_retrieval"


def score_row(row: dict[str, Any], *, threshold: float = RETRIEVAL_HIT_THRESHOLD) -> dict[str, Any]:
    """Compute the retrieval-recall fields and stage attribution for one result row."""
    stats = retrieval_recall(row.get("ground_truth"), row.get("kg_hop_path"), row.get("query") or "")
    outcome = attribute_outcome(
        row.get("judge_label"),
        row.get("kg_row_count"),
        stats["recall"],
        threshold=threshold,
    )
    return {
        "retrieval_recall": stats["recall"],
        "retrieval_tokens_wanted": stats["wanted"],
        "retrieval_tokens_found": stats["found"],
        "retrieval_missing_tokens": stats["missing"],
        "outcome": outcome,
    }
