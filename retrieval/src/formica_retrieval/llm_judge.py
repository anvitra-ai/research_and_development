"""LLM-as-judge relevance evaluation for query/summary pairs.

Reads a batch-pipeline results JSONL (query + summary columns), judges each pair
against a fixed relevance rubric using Gemini, and writes index/query/summary/
label/rationale to CSV and JSONL. Rows with an empty summary (e.g. error:
"no_entities_resolved" -- nothing was ever produced to judge) are auto-labeled
"Not relevant" without spending an LLM call, matching how this rubric is meant
to be applied.

Usage:
  python3 retrieval/llm_judge.py \
    --input data/banking_pipeline_results_with_summary.jsonl \
    --output-csv data/llm_judge_results.csv \
    --output-jsonl data/llm_judge_results.jsonl \
    --resume
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel

from paths import DATA_DIR, RD_ROOT

for _env_path in (RD_ROOT / ".env", RD_ROOT / "graphiti" / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path)
        break

import os

DEFAULT_INPUT = DATA_DIR / "banking_pipeline_results_with_summary.jsonl"
DEFAULT_OUTPUT_CSV = DATA_DIR / "llm_judge_results.csv"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "llm_judge_results.jsonl"
JUDGE_MODEL = "gemini-3.5-flash-lite"

RUBRIC_TEMPLATE = """You are an expert answer-relevance evaluator for AI conversations. You will receive a user request(query) and an assistant output(summary). Classify how well the assistant output addresses the user request. The question and answers are relevant to market scenarios.

## Scope
- Judge relevance, topical alignment, completeness, and whether the response resolves the request.
- Assess the assistant output against the user request only; do not infer unstated requirements.
- Do not judge writing style or factual correctness unless it prevents the output from meaningfully answering the request.

## Golden Rule
Select Relevant only when the output directly addresses the user's primary request, covers all material parts, and stays on topic.

## Labels
- Relevant: directly addresses the primary request, covers all material parts, and remains on topic.
- Somewhat relevant: addresses part of the request but misses a material requirement, key constraint, or necessary next step.
- Not relevant: is off-topic, evasive, unrelated, or fails to address the primary request.

## Decision Rules
1. Identify the user's primary goal and any material sub-requests or constraints.
2. Compare the assistant output against those requirements.
3. If multiple requests exist, assess whether the output resolves the primary request and all material parts.
4. Choose exactly one label. When evidence is ambiguous, choose the lower-supported label rather than assuming unstated coverage.

## Examples
- Input: "How do I reset my password?" Output: "Use the Forgot password link on the sign-in page." -> Relevant
- Input: "Compare the Pro and Team plans." Output: "The Pro plan includes analytics." -> Somewhat relevant
- Input: "How do I export a CSV?" Output: "Our product is designed for collaboration." -> Not relevant

User request: {query}
Assistant output: {summary}

Respond with a label (Relevant, Somewhat relevant, or Not relevant) and a one-sentence rationale."""


class JudgeVerdict(BaseModel):
    label: Literal["Relevant", "Somewhat relevant", "Not relevant"]
    rationale: str


_TRANSIENT_ERRORS = (ServerError,)


def judge_pair(client: genai.Client, query: str, summary: str, max_retries: int = 4) -> JudgeVerdict:
    prompt = RUBRIC_TEMPLATE.format(query=query, summary=summary)
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeVerdict,
                ),
            )
            return JudgeVerdict.model_validate_json(response.text)
        except _TRANSIENT_ERRORS as e:
            if attempt == max_retries:
                raise
            delay = 5 * (2**attempt)
            print(f"  transient error (attempt {attempt + 1}/{max_retries + 1}): {e!r} -- retrying in {delay}s")
            time.sleep(delay)


def load_done_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["index"])
    return done


def append_result(result: dict, csv_path: Path, jsonl_path: Path, *, write_header: bool) -> None:
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=write_header, index=False)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge relevance evaluation.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--resume", action="store_true", help="Skip indices already in the output.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    client = genai.Client(api_key=api_key)

    rows = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"index": i, "query": r.get("query", ""), "summary": r.get("summary") or ""})
    if args.limit:
        rows = rows[: args.limit]

    csv_path = Path(args.output_csv)
    jsonl_path = Path(args.output_jsonl)
    done = load_done_indices(jsonl_path) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} indices already judged")
    write_header = not csv_path.exists() or not args.resume

    llm_calls = 0
    auto_labeled = 0
    for row in rows:
        if row["index"] in done:
            continue

        if not row["summary"].strip():
            # Nothing was ever produced to judge (e.g. no_entities_resolved) --
            # auto-label without spending an LLM call.
            result = {
                "index": row["index"],
                "query": row["query"],
                "summary": row["summary"],
                "label": "Not relevant",
                "rationale": "No summary was produced for this query (empty assistant output).",
            }
            auto_labeled += 1
        else:
            verdict = judge_pair(client, row["query"], row["summary"])
            result = {
                "index": row["index"],
                "query": row["query"],
                "summary": row["summary"],
                "label": verdict.label,
                "rationale": verdict.rationale,
            }
            llm_calls += 1

        append_result(result, csv_path, jsonl_path, write_header=write_header)
        write_header = False

        done_so_far = len(done) + llm_calls + auto_labeled
        if done_so_far % 50 == 0:
            print(f"  {done_so_far}/{len(rows)} judged ({llm_calls} LLM calls, {auto_labeled} auto-labeled)")

    print(f"\nDone: {llm_calls} LLM judge calls, {auto_labeled} auto-labeled (empty summary)")

    final = pd.read_csv(csv_path)
    print(f"\nSaved {len(final)} results to {csv_path}")
    counts = final["label"].value_counts()
    total = len(final)
    for label, count in counts.items():
        print(f"  {label}: {count} ({count / total * 100:.0f}%)")


if __name__ == "__main__":
    main()
