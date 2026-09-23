"""Formica template resolution: slots in, Cypher rows out.

This module is now a facade. The implementation was split when it passed 1500
lines and three unrelated concerns had grown into each other:

  slot_extraction.py     which entities fill nnp1/nnp2/prop1, and which mode
  template_expansion.py  slots -> a runnable Cypher string + parameters
  template_execution.py  run it, with the fallback cascade and aggregation
  row_values.py          reading a row's numeric value and segment

Everything the old module exported is re-exported here, so existing imports
(`from .template_resolver import expand_formica_template`) keep working.
"""

from __future__ import annotations

from .row_values import (  # noqa: F401
    _numeric_value_from_fact,
    _segment_of,
    _usable_value,
)
from .slot_extraction import (  # noqa: F401
    DEFAULT_TEMPLATE_PATH,
    extract_triplet_slots,
    infer_prop1,
    infer_prop1_list,
    is_cross_company_query,
    load_templates,
    resolve_segment_ids,
    resolve_segment_names,
)
from .template_expansion import (  # noqa: F401
    _apply_compare_fallback,
    _relaxed_simple_cypher,
    expand_formica_template,
)
from .template_execution import execute_formica_template  # noqa: F401

# Previously re-exported from here for callers' convenience; kept so imports
# that reached through this module for them do not break.
from .entity_resolver import enrich_for_formica_template  # noqa: F401
from .fact_search import (  # noqa: F401
    semantic_fact_search,
    semantic_relation_names,
    text_search_fallback,
)
