"""FastAPI service wrapping the Formica KG query pipeline for ad-hoc search.

Loads NER/embedder/Neo4j driver/Gemini client once at startup (all of
process_query()'s heavy resources), then answers each request by running the same
pipeline used by scripts/run_batch_pipeline.py: entity resolution -> template
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

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from formica_retrieval.pipeline import PipelineResources, TRAIN_CSV, load_resources, process_query

# Populated once at startup by the lifespan handler below; None until then (or if
# Gemini/Neo4j failed to initialize), which /search and /health both check for.
_resources: PipelineResources | None = None
_startup_error: str | None = None

# query text -> ground_truth, loaded once from the same labeled dataset the batch
# pipeline uses (indian_banks_993_queries.csv). Lets an ad-hoc /search request for
# a query that happens to be one of the labeled ones get the same grounded judge
# (compared against a reference answer) that batch runs get, instead of always
# falling back to the ungrounded rubric -- and surfaces the reference answer
# itself in the response for the caller to compare against.
_ground_truth_by_query: dict[str, str] = {}


def _load_ground_truth() -> dict[str, str]:
    if not Path(TRAIN_CSV).exists():
        return {}
    df = pd.read_csv(TRAIN_CSV)
    if "ground_truth" not in df.columns:
        return {}
    return {
        str(row["text"]): str(row["ground_truth"])
        for _, row in df.iterrows()
        if pd.notna(row.get("ground_truth"))
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _resources, _startup_error, _ground_truth_by_query
    print("Loading pipeline resources (NER, embedder, Neo4j driver, Gemini client)...")
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
    _ground_truth_by_query = _load_ground_truth()
    print(f"Loaded ground truth for {len(_ground_truth_by_query)} known queries.")
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
    ground_truth: str | None = None
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

    # No gold label exists for an ad-hoc query: it is a HUMAN-assigned template
    # class, used only to score the router in batch runs. Passing the router's
    # own output here made template_match compare the router against itself and
    # report True unconditionally. Empty means "unknown", which process_query
    # turns into template_match=None -- the honest answer.
    ground_truth = _ground_truth_by_query.get(body.query)
    result = process_query(
        body.query,
        "",
        _resources,
        summarize=want_summary,
        judge=body.judge,
        ground_truth=ground_truth,
    )
    return SearchResult(**{k: result.get(k) for k in SearchResult.model_fields})
