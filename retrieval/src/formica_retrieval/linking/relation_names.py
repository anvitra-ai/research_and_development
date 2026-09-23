"""Canonicalising Graphiti's inconsistent relation-name spellings.

Graphiti writes the same edge_types class under more than one spelling in a
single graph (MetricObservation and METRIC_OBSERVATION, ExposureRelation and
EXPOSURE_RELATION, ...). Retrieval filters on r.name, so a filter carrying only
one spelling silently misses every edge that got the other.

Its own module because both the slot/Cypher layer and the fact-search layer need
it, and because scripts/check_graph_contract.py uses it to REPORT the drift --
the long-term fix is normalising at write time, at which point this compensation
can be deleted.
"""

from __future__ import annotations

import re


def _screaming_snake(name: str) -> str | None:
    """"MetricObservation" -> "METRIC_OBSERVATION"; None if not PascalCase."""
    if "_" in name or not any(c.islower() for c in name):
        return None
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def _with_casing_variants(relation_names: list[str]) -> list[str]:
    """Add each relation name's SCREAMING_SNAKE twin to the list.

    Graphiti's extraction is not consistent about how it spells an edge_types
    class name into r.name: the same ontology class lands as "MetricObservation"
    for most episodes but "METRIC_OBSERVATION" for others, in the SAME graph
    (confirmed after a full re-ingest: 456 MetricObservation vs 20
    METRIC_OBSERVATION, and likewise for ExposureRelation/EXPOSURE_RELATION,
    OwnershipStake/OWNERSHIP_STAKE, SupportedByRelation/SUPPORTED_BY_RELATION,
    GovernanceRole/GOVERNANCE_ROLE, CorporateStructureRelation/...). A prop1_list
    carrying only the PascalCase spelling silently misses every fact that got the
    other one -- which is exactly what dropped ALL of Axis Bank's metrics out of
    metric-comparison queries. Matching both spellings costs nothing and
    survives whatever casing the next re-ingest happens to produce.
    """
    out = list(relation_names)
    for name in relation_names:
        variant = _screaming_snake(name)
        if variant and variant not in out:
            out.append(variant)
    return out
