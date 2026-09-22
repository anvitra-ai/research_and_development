"""Heavy pipeline resources: NER/embedder/classifier/Neo4j/Gemini, loaded once.

`load_resources()` is the single entry point -- called once per process (a batch
run, the FastAPI service's startup, a REPL session), returning a `PipelineResources`
bundle that every `process_query()` call reuses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from google import genai
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from . import config as _cfg
from .config import GEMINI_API_KEY, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .entity_resolver import load_aliases
from .paths import MODELS_DIR
from .template_classifier import FormicaTemplateClassifier

FORMICA_MODEL = MODELS_DIR / "formica-template-classifier.joblib"
DEFAULT_NER_MODEL = _cfg.NER_MODEL
EMBEDDING_MODEL = _cfg.EMBEDDING_MODEL


@dataclass
class PipelineResources:
    """Heavy objects loaded once and reused across all queries in a batch run."""

    ner: Any
    embedder: SentenceTransformer
    classifier: FormicaTemplateClassifier
    driver: Any
    node_df: pd.DataFrame
    node_emb: Any
    aliases: dict[str, dict[str, str]]
    gemini: genai.Client | None = None
    # Built lazily by load_fact_index() the first time the semantic fallback
    # actually needs it -- most queries resolve entities normally and never
    # touch it, so embedding every fact in the graph at startup would be paid
    # for nothing on the common path.
    fact_index: Any = None


def _load_ner_pipeline(model_name: str):
    return pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        device=-1,
    )


def load_ner_pipeline(model_name: str | None = None):
    """Load NER pipeline with optional env override and fallback."""
    ner_model = model_name or os.getenv("NER_MODEL", DEFAULT_NER_MODEL)
    try:
        ner = _load_ner_pipeline(ner_model)
        if ner_model != DEFAULT_NER_MODEL:
            print(f"NER model: {ner_model}")
        return ner
    except (ValueError, OSError, AttributeError) as exc:
        if ner_model == DEFAULT_NER_MODEL:
            raise
        print(
            f"Warning: failed to load NER model '{ner_model}' ({exc}). "
            f"Falling back to {DEFAULT_NER_MODEL}."
        )
        return _load_ner_pipeline(DEFAULT_NER_MODEL)


# Graphiti entity-type labels (PascalCase, real Neo4j labels) that correspond to a
# legacy Formica type bucket the existing templates/vocabulary already know about.
# Anything not listed here falls back to the label itself, upper-cased -- still
# correctly indexed for gazetteer/alias/semantic matching, just not specially
# prioritized by the legacy SLOT_SUBJECT_PRIORITY / FORMICA_NEEDED_TYPES sets
# (which were built for kg/india_theme_kg.cypher's thematic graph, not banking).
_GRAPHITI_TO_LEGACY_TYPE: dict[str, str] = {
    "Company": "COMPANY",
    "Geography": "GEOGRAPHY",
    "Sector": "SECTOR",
    "Product": "PRODUCT",
    "MacroeconomicFactor": "MACRO_VAR",
}


def load_node_index(driver) -> tuple[pd.DataFrame, Any]:
    """Load all KG nodes from Neo4j and pre-compute embedding vectors for semantic linking.

    Reads Graphiti's actual node schema: id = n.uuid, name = n.name, and node_type is
    derived from the node's Neo4j labels (every Graphiti node carries a generic
    "Entity" label plus one specific type label, e.g. ["Entity", "Company"] -- the
    specific one is used, mapped to a legacy type bucket where one exists (see
    _GRAPHITI_TO_LEGACY_TYPE), otherwise upper-cased as-is.
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    with driver.session() as session:
        nodes = session.execute_read(
            lambda tx: list(
                tx.run(
                    """
                    MATCH (n:Entity)
                    RETURN n.uuid AS id, n.name AS name, labels(n) AS node_labels
                    """
                )
            )
        )

    def _resolve_type(node_labels: list[str] | None) -> str | None:
        specific = [l for l in (node_labels or []) if l != "Entity"]
        if not specific:
            return None
        label = specific[0]
        return _GRAPHITI_TO_LEGACY_TYPE.get(label, label.upper())

    node_df = (
        pd.DataFrame(
            [
                {"id": r["id"], "node_name": r["name"], "node_type": _resolve_type(r["node_labels"])}
                for r in nodes
            ]
        )
        .dropna(subset=["node_name"])
        .drop_duplicates(subset=["node_name"])
        .reset_index(drop=True)
    )
    node_names = node_df["node_name"].astype(str).tolist()
    node_emb = embedder.encode(node_names, convert_to_tensor=True, normalize_embeddings=True)
    return embedder, node_df, node_emb


_FACT_INDEX_CYPHER = (
    "MATCH (c:Company)-[r:RELATES_TO]-(o:Entity)\n"
    "WHERE r.fact IS NOT NULL\n"
    "RETURN DISTINCT c.name AS subject_name, r.name AS relationship, o.name AS object_name, "
    "r.fact AS fact, r.valid_at AS valid_at"
)


def load_fact_index(driver, embedder) -> tuple[list[dict[str, Any]], Any]:
    """Embed every company fact in the graph for semantic fallback ranking.

    Same shape as load_node_index above, but over relationship fact TEXT rather
    than node names: the fallback that uses this answers cross-company questions
    that name no resolvable entity at all, so it has to search what the facts
    actually SAY, not what they're linked to.
    """
    with driver.session() as session:
        rows = [dict(r) for r in session.run(_FACT_INDEX_CYPHER)]
    if not rows:
        return [], None
    fact_emb = embedder.encode(
        [str(r["fact"]) for r in rows], convert_to_tensor=True, normalize_embeddings=True
    )
    return rows, fact_emb


def load_resources(summarize: bool = False, judge: bool = False) -> PipelineResources:
    """Initialize NER, embedder, classifier, Neo4j driver, and optional Gemini client.

    The same Gemini client is reused for both summarization and LLM-as-judge, so
    it's created whenever either is requested.
    """
    ner = load_ner_pipeline()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    embedder, node_df, node_emb = load_node_index(driver)
    classifier = FormicaTemplateClassifier(model_path=FORMICA_MODEL)
    classifier.load()
    aliases = load_aliases()
    gemini = None
    if summarize or judge:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when summarization or judging is enabled")
        gemini = genai.Client(api_key=GEMINI_API_KEY)
    return PipelineResources(
        ner=ner,
        embedder=embedder,
        classifier=classifier,
        driver=driver,
        node_df=node_df,
        node_emb=node_emb,
        aliases=aliases,
        gemini=gemini,
    )
