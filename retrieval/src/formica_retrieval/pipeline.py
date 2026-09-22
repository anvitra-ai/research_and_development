"""Shared Formica KG query pipeline.

End-to-end flow for each natural-language question:

  1. Entity resolution (first pass)
     Link query spans to KG nodes via gazetteer, aliases, NER + semantic search.

  2. Template classification
     Mask resolved entities as <TYPE> tags and predict a Formica template class
     (F_Simple, F_QuantCount, etc.) using the trained SVM classifier.

  3. Entity resolution (second pass)
     Re-link with template-aware type preferences so slots get the right node types.

  4. Entity enrichment
     Pull in additional gazetteer/alias hits needed by the predicted template.

  5. Template expansion
     Map entities to triplet slots (nnp1, nnp2, prop1, …) and build Cypher.

  6. Cypher execution
     Run against Neo4j with fallbacks (propagation, transit, beneficiaries, …).

  7. Optional summarization
     Format KG rows as hop text and summarize with Gemini.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from paths import DATA_DIR, MODELS_DIR, RD_ROOT

from entity_resolver import enrich_for_formica_template, load_aliases, resolve_entities
from formica_template_classifier import FormicaTemplateClassifier, rule_label_formica
from formica_template_resolver import execute_formica_template, expand_formica_template
from llm_judge import judge_pair

# GEMINI_API_KEY lives in a .env file, not the shell environment -- load it the same
# way the graphiti notebook does. Prefer a .env at the research_and_development root
# (shared across subprojects); fall back to graphiti/.env, the only place the key
# currently lives, so it doesn't need to be duplicated. Either load is a no-op if the
# variable is already set in the actual environment.
for _env_path in (RD_ROOT / ".env", RD_ROOT / "graphiti" / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path)
        break

TRAIN_CSV = DATA_DIR / "banking_queries/indian_banks_1200_queries.csv"
LEGACY_TRAIN_CSV = DATA_DIR / "deberta_stock_impact_train.csv"
DEFAULT_OUTPUT_CSV = DATA_DIR / "formica_pipeline_results.csv"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "formica_pipeline_results.jsonl"
FORMICA_MODEL = MODELS_DIR / "formica-template-classifier.joblib"
DEFAULT_NER_MODEL = "dslim/bert-base-NER"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin1234")

RESULT_FIELDS = (
    "query",
    "gold_label",
    "predicted_template",
    "template_confidence",
    "template_id",
    "template_match",
    "entities",
    "resolved_entities",
    "masked_query",
    "cypher_parameters",
    "triplet_slots",
    "cypher_strategy",
    "kg_row_count",
    "kg_hop_path",
    "summary",
    "judge_label",
    "judge_rationale",
    "error",
)


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


def mask_entities_with_types(text: str, entity_types: list[tuple[str, str]]) -> str:
    """Replace entity spans with <TYPE> tags for the Formica classifier."""
    masked = text
    for entity, entity_type in sorted(entity_types, key=lambda x: len(x[0]), reverse=True):
        if not entity or not entity_type:
            continue
        tag = f"<{entity_type.upper()}>"
        pattern = re.compile(re.escape(entity), flags=re.IGNORECASE)
        masked = pattern.sub(tag, masked)
    return masked


def entity_types_from_matches(matches_df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(row["entity"], row["matched_type"]) for _, row in matches_df.iterrows()]


def format_kg_rows_for_summary(rows: list, expanded: dict[str, Any]) -> str:
    """Turn Neo4j result rows into readable hop/path text for Gemini summarization.

    Prefers Graphiti's own stored text over reconstructing a sentence from raw
    subject/relationship/object triples, since it's already a precise, human-written
    summary of that specific fact:
      - attribute_lookup rows (a node's own property, not an edge) -> "X's <attr> is <value>"
      - a single edge's r.fact -> the fact text verbatim
      - aggregate/list facts (F_QuantCount's collect(), F_CompMore/Less/Approx's path
        facts_a/facts_b) -> the count/comparison header plus every underlying fact
    Falls through to the legacy path/transit formatting, then the generic key=value
    reconstruction, for rows that carry none of the above (e.g. the still-unfixed
    legacy :Node-schema special modes, which have no fact/attribute fields at all).
    """
    if not rows:
        return ""
    lines = [
        f"Template: {expanded['template_id']} ({expanded['template_label']})",
        f"Query intent: {expanded['description']}",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        data = dict(row)
        if "attribute" in data and "value" in data:
            subject = data.get("subject_name", "?")
            lines.append(f"Row {i}: {subject}'s {data['attribute']} is {data['value']}")
        elif data.get("fact"):
            lines.append(f"Row {i}: {data['fact']}")
        elif data.get("facts"):
            facts = [f for f in data["facts"] if f]
            header_parts = [f"{k}={v}" for k, v in data.items() if v is not None and k != "facts"]
            lines.append(f"Row {i}: " + " | ".join(header_parts) + f" ({len(facts)} facts)")
            for f in facts:
                lines.append(f"  - {f}")
        elif data.get("facts_a") or data.get("facts_b"):
            header_parts = [
                f"{k}={v}" for k, v in data.items() if v is not None and k not in ("facts_a", "facts_b")
            ]
            lines.append(f"Row {i}: " + " | ".join(header_parts))
            for label, facts in (("a", data.get("facts_a")), ("b", data.get("facts_b"))):
                for f in facts or []:
                    if f:
                        lines.append(f"  [{label}] {f}")
        elif "path_names" in data and "rel_types" in data:
            names = data.get("path_names") or []
            rels = data.get("rel_types") or []
            header = (
                f"Path {i}: {data.get('source_name', names[0] if names else '?')}"
                f" -> {data.get('target_name', names[-1] if names else '?')}"
                f" ({data.get('hops', len(rels))} hops)"
            )
            lines.append(header)
            for j, rel in enumerate(rels):
                left = names[j] if j < len(names) else "?"
                right = names[j + 1] if j + 1 < len(names) else "?"
                lines.append(f"  {left} -[{rel}]-> {right}")
        elif "commodity_name" in data and ("geography_name" in data or "chokepoint" in data):
            geo = data.get("geography_name") or data.get("chokepoint")
            share = f" share={data['share']}" if data.get("share") is not None else ""
            extra = f" [{data['channel']}]" if data.get("channel") else ""
            lines.append(
                f"Row {i}: {data['commodity_name']} -[TRANSITS]-> {geo}{share}{extra}"
            )
        else:
            parts = [f"{k}={v}" for k, v in data.items() if v is not None]
            lines.append(f"Row {i}: " + " | ".join(parts))
    return "\n".join(lines)


def summarize_hops(client: genai.Client, hop_path: str) -> str:
    prompt = f"""
You are a financial analyst answering a business question using knowledge-graph facts.

Write a plain-language summary of what these facts actually say, in business terms.

Requirements:
1. Focus only on the business/financial substance -- never describe the graph
   itself (no "starting entity", "ending entity", "relationship", "path", or any
   other entity/graph-structure framing, and no numbered or labeled sections).
2. Write flowing prose, not a list.
3. Include specific numbers, dates, or ratios only if they appear in the facts
   below; never invent or estimate them.
4. Maximum 2 sentences.

Knowledge graph facts:

{hop_path}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )
    return response.text


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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when summarization or judging is enabled")
        gemini = genai.Client(api_key=api_key)
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


def empty_query_result(query: str, gold_label: str) -> dict[str, Any]:
    return {field: None for field in RESULT_FIELDS} | {
        "query": query,
        "gold_label": gold_label,
        "kg_row_count": 0,
    }


def gold_label_from_row(row: pd.Series) -> str:
    if "formica_label" in row and pd.notna(row["formica_label"]):
        return str(row["formica_label"])
    domain = str(row.get("label", "")) or None
    return rule_label_formica(str(row["text"]), domain)


def _finish(
    result: dict[str, Any], query: str, resources: PipelineResources, judge: bool
) -> dict[str, Any]:
    """Apply the optional LLM-as-judge step, then return the result.

    Every process_query() return funnels through here, so a query that fails
    entity resolution or Cypher execution (empty summary) still gets auto-labeled
    the same way llm_judge.py's standalone batch script does, instead of only
    judging the success path.
    """
    if judge:
        summary = (result.get("summary") or "").strip()
        if not summary:
            result["judge_label"] = "Not relevant"
            result["judge_rationale"] = "No summary was produced for this query (empty assistant output)."
        else:
            verdict = judge_pair(resources.gemini, query, summary)
            result["judge_label"] = verdict.label
            result["judge_rationale"] = verdict.rationale
    return result


def process_query(
    query: str,
    gold_label: str,
    resources: PipelineResources,
    *,
    summarize: bool = False,
    judge: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline for a single query and return a result dict."""
    result = empty_query_result(query, gold_label)

    try:
        # --- Step 1: first-pass entity linking (no template context yet) ---
        _, matches_df = resolve_entities(
            query,
            resources.ner,
            resources.embedder,
            resources.node_df,
            resources.node_emb,
            aliases=resources.aliases,
        )
        if matches_df.empty:
            result["error"] = "no_entities_resolved"
            return _finish(result, query, resources, judge)

        masked_query = mask_entities_with_types(query, entity_types_from_matches(matches_df))

        # --- Step 2: classify masked query into a Formica template ---
        template_label, confidence = resources.classifier.predict(masked_query)

        # --- Step 3: second-pass linking with template-aware type preferences ---
        _, matches_df = resolve_entities(
            query,
            resources.ner,
            resources.embedder,
            resources.node_df,
            resources.node_emb,
            aliases=resources.aliases,
            template_label=template_label,
        )
        result["entities"] = matches_df["entity"].tolist() if not matches_df.empty else None
        result["masked_query"] = (
            mask_entities_with_types(query, entity_types_from_matches(matches_df))
            if not matches_df.empty
            else masked_query
        )
        result["predicted_template"] = template_label
        result["template_confidence"] = float(confidence)
        result["template_match"] = template_label == gold_label

        if matches_df.empty:
            result["error"] = "no_entities_resolved"
            return _finish(result, query, resources, judge)

        # --- Step 4: add missing entities the template expects (gazetteer/alias sweep) ---
        matches_df = enrich_for_formica_template(
            query, template_label, matches_df, resources.node_df, aliases=resources.aliases
        )
        result["resolved_entities"] = matches_df.to_dict(orient="records")

        # --- Step 5: fill triplet slots and build parameterized Cypher ---
        expanded = expand_formica_template(query, template_label, matches_df)
        result["template_id"] = expanded["template_id"]
        result["cypher_parameters"] = expanded.get("parameters")
        result["triplet_slots"] = expanded.get("triplet_slots")

        # --- Step 6: execute Cypher (with retries when primary query returns no rows) ---
        kg_rows, strategy = execute_formica_template(resources.driver, expanded)
        if not kg_rows and expanded.get("missing_parameters"):
            result["error"] = f"missing_parameters:{expanded['missing_parameters']}"
            return _finish(result, query, resources, judge)

        result["cypher_strategy"] = strategy
        result["kg_row_count"] = len(kg_rows)
        kg_hop_path = format_kg_rows_for_summary(kg_rows, expanded)
        result["kg_hop_path"] = kg_hop_path or None

        # --- Step 7 (optional): natural-language summary of KG paths ---
        if summarize and kg_hop_path and resources.gemini is not None:
            result["summary"] = summarize_hops(resources.gemini, kg_hop_path)

    except Exception as exc:
        result["error"] = str(exc)

    return _finish(result, query, resources, judge)


def serialize_result_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    json_fields = ("entities", "resolved_entities", "cypher_parameters", "triplet_slots")
    row = dict(result)
    for field in json_fields:
        if row.get(field) is not None:
            row[field] = json.dumps(row[field])
    return row


def append_result(
    result: dict[str, Any],
    csv_path: Path,
    jsonl_path: Path,
    *,
    write_header: bool,
) -> None:
    pd.DataFrame([serialize_result_for_csv(result)]).to_csv(
        csv_path, mode="a", header=write_header, index=False
    )
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_summary_stats(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    final = pd.read_csv(csv_path)
    acc = None
    if "template_match" in final:
        acc = final["template_match"].astype(str).str.lower().eq("true").mean()
    kg_hit = None
    if "kg_row_count" in final:
        kg_hit = (pd.to_numeric(final["kg_row_count"], errors="coerce").fillna(0) > 0).mean()
    print(f"\nSaved {len(final)} results to {csv_path}")
    if acc is not None:
        print(f"Template accuracy: {acc:.3f}")
    if kg_hit is not None:
        print(f"KG row hit rate: {kg_hit:.3f}")
    if "judge_label" in final and final["judge_label"].notna().any():
        counts = final["judge_label"].value_counts()
        total = counts.sum()
        print("\nLLM-judge relevance:")
        for label, count in counts.items():
            print(f"  {label}: {count} ({count / total * 100:.0f}%)")
