"""FastAPI service wrapping the Formica KG query pipeline for ad-hoc search.

Loads NER/embedder/classifier/Neo4j driver/Gemini client once at startup (all of
process_query()'s heavy resources), then answers each request by running the same
pipeline used by run_batch_pipeline.py: entity resolution -> template
classification -> Cypher execution -> optional summarization -> optional
LLM-as-judge.

Run:
  cd research_and_development/retrieval
  .venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000

Then:
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query": "What is State Bank of India'"'"'s ticker symbol?", "summarize": true}'
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_RETRIEVAL_ROOT = Path(__file__).resolve().parent
if str(_RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from formica_pipeline import PipelineResources, load_resources, process_query
from formica_template_classifier import rule_label_formica

# Populated once at startup by the lifespan handler below; None until then (or if
# Gemini/Neo4j failed to initialize), which /search and /health both check for.
_resources: PipelineResources | None = None
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _resources, _startup_error
    print("Loading pipeline resources (NER, embedder, classifier, Neo4j driver, Gemini client)...")
    try:
        # Always request both -- creating the Gemini client is cheap (no API call
        # happens until a request actually asks for summarize/judge); this just
        # means a single request can toggle either on without reloading resources.
        _resources = load_resources(summarize=True, judge=True)
        print("Resources loaded. Ready to serve.")
    except Exception as exc:
        # Missing GEMINI_API_KEY, unreachable Neo4j, etc. -- start anyway so
        # /health reports the real problem instead of the process refusing to boot.
        _startup_error = str(exc)
        print(f"WARNING: resource loading failed, /search will 503 until fixed: {exc}")
    yield
    if _resources is not None:
        _resources.driver.close()


app = FastAPI(
    title="Formica KG Search",
    description="Natural-language search over the Graphiti banking knowledge graph.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question to search the KG with.")
    summarize: bool = Field(False, description="Generate a Gemini natural-language summary of the KG result.")
    judge: bool = Field(False, description="LLM-as-judge the summary for relevance (implies summarize).")


class SearchResult(BaseModel):
    query: str
    predicted_template: str | None = None
    template_confidence: float | None = None
    entities: list[str] | None = None
    masked_query: str | None = None
    resolved_entities: list[dict[str, Any]] | None = None
    cypher_strategy: str | None = None
    kg_row_count: int = 0
    kg_hop_path: str | None = None
    summary: str | None = None
    judge_label: str | None = None
    judge_rationale: str | None = None
    error: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    if _resources is not None:
        return {"status": "ok"}
    return {"status": "unavailable", "error": _startup_error or "resources not loaded"}


@app.post("/search", response_model=SearchResult)
def search(body: SearchRequest) -> SearchResult:
    if _resources is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Resources not loaded")

    want_summary = body.summarize or body.judge
    if want_summary and _resources.gemini is None:
        raise HTTPException(
            status_code=400,
            detail="summarize/judge requested but GEMINI_API_KEY was not configured at startup",
        )

    gold_label = rule_label_formica(body.query)
    result = process_query(body.query, gold_label, _resources, summarize=want_summary, judge=body.judge)
    return SearchResult(**{k: result.get(k) for k in SearchResult.model_fields})
