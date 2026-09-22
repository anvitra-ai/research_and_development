"""Centralized runtime configuration: env loading + Neo4j/Gemini credentials.

Every module that needs a Neo4j driver or a Gemini client should read its
settings from here instead of hardcoding a URI/password or re-implementing
`load_dotenv()` -- keeps credential handling in exactly one place.

Values come from the environment (a `.env` file in the repo root or
`retrieval/` is loaded automatically via python-dotenv). The bundled defaults
are for local development only and are safe to keep in source; override them
via real environment variables for anything beyond a local machine.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .paths import REPO_ROOT, RETRIEVAL_ROOT

# Load from retrieval/.env first (this package's own settings), then fall back
# to a repo-root .env if present. Neither call overrides a variable that's
# already set in the real environment.
load_dotenv(RETRIEVAL_ROOT / ".env")
load_dotenv(REPO_ROOT / ".env")

def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to the in-code default.

    Deliberately tolerant: a malformed value falls back rather than crashing the
    process at import time, since these are tuning knobs and a typo in a .env
    should not take down a 2.5-hour benchmark run.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Warning: {name}={raw!r} is not an int; using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Warning: {name}={raw!r} is not a float; using default {default}")
        return default


# --- Connections and credentials -------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "admin1234")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Models ----------------------------------------------------------------
# Summarisation runs on the stronger reasoning model; judging on the cheap fast
# one. They are separate settings on purpose -- the judge is a measuring
# instrument and changing it silently invalidates comparisons against earlier
# benchmark runs, so it should move independently of the summariser.
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemini-3.1-pro-preview")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
NER_MODEL = os.getenv("NER_MODEL", "dslim/bert-base-NER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Both generation ends are greedy so that a score change is attributable to a
# pipeline change rather than to sampling. Re-judging an identical results file
# at the API default moved the Relevant count by ~15 rows.
SUMMARY_TEMPERATURE = _env_float("SUMMARY_TEMPERATURE", 0.0)
JUDGE_TEMPERATURE = _env_float("JUDGE_TEMPERATURE", 0.0)
# GeminiClient's per-model default doesn't recognise the pro-preview name and
# silently caps output at 8192 tokens.
SUMMARY_MAX_TOKENS = _env_int("SUMMARY_MAX_TOKENS", 65536)

# Hard bound on each LLM call. Without one, a stalled connection blocks its
# worker thread indefinitely -- under concurrency that wedges the whole run
# (observed: every worker hung, 0 progress, unrecoverable without a kill).
SUMMARY_TIMEOUT_MS = _env_int("SUMMARY_TIMEOUT_MS", 120_000)
SUMMARY_MAX_RETRIES = _env_int("SUMMARY_MAX_RETRIES", 3)
JUDGE_TIMEOUT_MS = _env_int("JUDGE_TIMEOUT_MS", 60_000)

# --- Retrieval tuning ------------------------------------------------------
# A candidate row set whose best row scores below this is treated as off-topic
# and the cascade keeps looking. See relevance.py for how it is applied.
RELEVANCE_FLOOR = _env_float("RELEVANCE_FLOOR", 0.32)
# Row budgets. The default cap exists to stop a long tail of loosely-related
# facts burying the one row that holds the answer; the enumeration cap is
# deliberately larger because "which banks..." needs breadth to be answerable.
MAX_MERGED_ROWS = _env_int("MAX_MERGED_ROWS", 40)
MAX_ENUMERATION_ROWS = _env_int("MAX_ENUMERATION_ROWS", 90)
MAX_SUPPLEMENT_ROWS = _env_int("MAX_SUPPLEMENT_ROWS", 8)
SUPPLEMENT_MIN_SIMILARITY = _env_float("SUPPLEMENT_MIN_SIMILARITY", 0.35)
# SWOT/analytical questions need to synthesise across scattered facts, so they
# get a wider, lower-threshold net than a single-fact lookup.
ANALYTICAL_SUPPLEMENT_ROWS = _env_int("ANALYTICAL_SUPPLEMENT_ROWS", 30)
ANALYTICAL_MIN_SIMILARITY = _env_float("ANALYTICAL_MIN_SIMILARITY", 0.20)
# Below this many retrieved rows, a node's `summary` profile is appended.
PROFILE_ROW_THRESHOLD = _env_int("PROFILE_ROW_THRESHOLD", 25)

# --- Evaluation ------------------------------------------------------------
# Share of the ground truth's material tokens that must be present before a
# failure is blamed on generation rather than retrieval.
RETRIEVAL_HIT_THRESHOLD = _env_float("RETRIEVAL_HIT_THRESHOLD", 1.0)
