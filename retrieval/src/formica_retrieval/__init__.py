"""formica_retrieval: natural-language -> Neo4j knowledge-graph retrieval pipeline.

Entity linking -> Formica template classification -> Cypher execution ->
optional Gemini summarization -> optional LLM-as-judge.

See `research_and_development/retrieval/README.md` for the module map and
how to run the batch pipeline / API / verification scripts.
"""

from .paths import DATA_DIR, MODELS_DIR, RD_ROOT, REPO_ROOT, RETRIEVAL_ROOT

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "RD_ROOT",
    "REPO_ROOT",
    "RETRIEVAL_ROOT",
]
