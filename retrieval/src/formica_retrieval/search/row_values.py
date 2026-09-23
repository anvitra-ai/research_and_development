"""Interpreting the numeric and segment fields on a retrieved row.

Graphiti's extraction is uneven: the same metric edge has `r.value` populated for
some companies and null for others, in units that vary by episode, while the
number the question actually asks about may exist only inside `r.fact`. Segment
membership is unreliable in its own way. Both are resolved here, once, so every
aggregation mode in template_execution agrees on what a row's value and segment
are.
"""

from __future__ import annotations

import re
from typing import Any

from ..routing.template_classes import COMPANY_SEGMENT_FALLBACK

_FACT_NUMBER_RE = re.compile(r"(?<!\d)(-?\d[\d,]*\.?\d*)\s*%")


def _numeric_value_from_fact(fact: str | None) -> float | None:
    if not fact:
        return None
    match = _FACT_NUMBER_RE.search(fact)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _usable_value(raw: Any, fact: str | None) -> float | None:
    """The trustworthy numeric for a ratio/percentage row, or None.

    r.value's UNIT varies by episode even on the same-named object node: "net
    profit" holds an absolute amount in crore for some companies (HDFC: 19060.0)
    while the question asks about the Y/Y GROWTH PERCENTAGE, which appears only
    as text in r.fact ("up 5% year-on-year"). An amount in the hundreds or
    thousands cannot be a real percentage, so it is untrustworthy here and the
    percentage written in the fact text is preferred.

    When the fact has no percentage to recover either (IDBI Bank's "reported
    total advances of Rs 218,399 crore" has no growth rate at all), the
    implausible >100 raw value must NOT be kept: it would win a percentage
    ranking purely because a currency amount is a bigger number than any real
    percentage -- exactly backwards from what "highest growth" means. Returning
    None drops the row instead of ranking on an impossible value.
    """
    if raw is None or abs(raw) > 100:
        return _numeric_value_from_fact(fact)
    return raw


# The three ownership categories a segment question can actually be asking
# about. The graph's own BankingSegment tag is trusted only when it is one of
# these -- several episodes tagged operating-model instead ("universal bank"),
# which answers a different question.
_REAL_SEGMENTS = frozenset({"Public Sector Banks", "Private Sector Banks", "Small Finance Banks"})


def _segment_of(company: str | None, graph_segment: str | None) -> str | None:
    """Segment for a company: the graph's tag when usable, else the static table.

    COMPANY_SEGMENT_FALLBACK holds stable public RBI classifications and carries
    this for most of the dataset, where the graph has no useful tag at all.
    """
    if graph_segment in _REAL_SEGMENTS:
        return graph_segment
    return COMPANY_SEGMENT_FALLBACK.get(company)
