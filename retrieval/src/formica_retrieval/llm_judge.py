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
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel

from . import config as _cfg
from .config import GEMINI_API_KEY
from .paths import DATA_DIR

DEFAULT_INPUT = DATA_DIR / "banking_pipeline_results_with_summary.jsonl"
DEFAULT_OUTPUT_CSV = DATA_DIR / "llm_judge_results.csv"
DEFAULT_OUTPUT_JSONL = DATA_DIR / "llm_judge_results.jsonl"
JUDGE_MODEL = _cfg.JUDGE_MODEL

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

# Grounded variant: judges the assistant summary against a REFERENCE ANSWER
# (ground truth pulled directly from the source document for that query)
# instead of an abstract "did it address the request" rubric with nothing to
# check against. This is the more reliable judge whenever ground truth is
# available -- it catches a confident-sounding but WRONG summary (which the
# ungrounded rubric above has no way to detect, since it never compares
# against what the actual correct answer is) and, symmetrically, it stops
# penalizing a summary for omitting a detail the reference answer itself
# doesn't have (many ground-truth answers in this dataset explicitly say a
# figure "was not found" -- a summary that also can't provide it isn't wrong).
GROUNDED_RUBRIC_TEMPLATE = """You are an expert answer-grader for AI-generated financial summaries. You will receive a user question, a REFERENCE ANSWER (ground truth, drawn directly from the source document), and an ASSISTANT SUMMARY (produced by a separate system from a knowledge graph built off that same document). Judge how well the assistant summary conveys the same substance as the reference answer.

## Scope
- Compare the assistant summary against the reference answer's specific facts, numbers, names, and dates -- not against the question in the abstract.
- The assistant summary does not need to match the reference answer's wording, only its factual content for the material parts of the question.
- Do not penalize the summary for including EXTRA correct information beyond the reference answer, unless that extra information contradicts the reference answer or crowds out/replaces the core answer.
- Do not penalize minor phrasing, formatting, rounding, or citation-source differences.
- If the reference answer itself states that a detail is unknown, unconfirmed, or was not found, do not penalize the summary for also not providing that detail -- that is agreement, not a gap.
- The same applies when the reference answer simply OMITS part of what the question asked. The reference answer defines the full expected scope of a correct answer: if the question asks for two figures and the reference answer gives only one, then supplying that one figure IS the complete correct answer. A summary that provides it and explicitly notes the other figure is not available must be judged on the figure it provided, and must NOT be downgraded for the note. Only penalize a missing detail when the reference answer actually contains it.

## Labels
- Relevant: the summary states all (or nearly all) of the material facts that the reference answer CONTAINS, with no material contradiction. Judge coverage against the reference answer's content, never against the number of clauses in the question.
- Somewhat relevant: the summary captures some of the reference answer's material facts but misses, is vague about, or gets wrong a meaningful part of it.
- Not relevant: the summary does not convey the reference answer's substance at all -- it is off-topic, contradicts the reference answer, or claims information is unavailable when the reference answer shows it is actually known.

## Decision Rules
1. Identify the material facts in the reference answer (the specific figures, names, dates, and claims that answer the question).
2. Check whether the assistant summary states those same facts, in substance (not exact wording).
3. Apply the "reference answer itself says unknown" exception from Scope before penalizing a missing detail.
4. Choose exactly one label based on how much of the reference answer's material content the summary actually conveys.

User question: {query}
Reference answer (ground truth): {ground_truth}
Assistant summary: {summary}

Respond with a label (Relevant, Somewhat relevant, or Not relevant) and a one-sentence rationale that references the specific facts you compared."""


class JudgeVerdict(BaseModel):
    label: Literal["Relevant", "Somewhat relevant", "Not relevant"]
    rationale: str


# ServerError covers 5xx, but a hung connection surfaces as a timeout or
# transport error instead, and those were previously unretried AND unbounded.
_TRANSIENT_ERRORS = (ServerError, TimeoutError, ConnectionError, OSError)
JUDGE_TIMEOUT_MS = _cfg.JUDGE_TIMEOUT_MS

# The judge is a measuring instrument, so it must not move when the code under
# test doesn't. At the API default temperature, re-judging an IDENTICAL results
# file moved the Relevant count by ~15 rows (~1.5 points) run to run -- larger
# than most single fixes being evaluated, which makes real gains indistinguishable
# from sampling noise. Greedy decoding removes that.
JUDGE_TEMPERATURE = _cfg.JUDGE_TEMPERATURE

# Label ordering used to break a 3-way tie in majority voting: with three
# distinct votes there is no majority, so take the middle (most conservative
# defensible) label rather than an arbitrary one.
_LABEL_RANK = {"Not relevant": 0, "Somewhat relevant": 1, "Relevant": 2}


def _judge_once(
    client: genai.Client,
    prompt: str,
    max_retries: int = 4,
) -> JudgeVerdict:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeVerdict,
                    temperature=JUDGE_TEMPERATURE,
                    # Without this the call can block its worker thread forever
                    # on a stalled connection -- see the note in summarization.py
                    # about a run wedging at 530/993 with every worker hung.
                    http_options=types.HttpOptions(timeout=JUDGE_TIMEOUT_MS),
                ),
            )
            return JudgeVerdict.model_validate_json(response.text)
        except _TRANSIENT_ERRORS as e:
            if attempt == max_retries:
                raise
            delay = 5 * (2**attempt)
            print(f"  transient error (attempt {attempt + 1}/{max_retries + 1}): {e!r} -- retrying in {delay}s")
            time.sleep(delay)


def judge_pair(
    client: genai.Client,
    query: str,
    summary: str,
    ground_truth: str | None = None,
    max_retries: int = 4,
    votes: int = 1,
) -> JudgeVerdict:
    """Judge a query/summary pair. Uses the grounded rubric (comparing against
    a reference answer) whenever ground_truth is given; falls back to the
    ungrounded rubric otherwise.

    Decoding is greedy (see JUDGE_TEMPERATURE), which is enough to make repeat
    runs stable. `votes` > 1 additionally takes a majority over that many calls
    -- greedy decoding is not a hard determinism guarantee on a hosted model, so
    this is available for runs where the measurement has to be trusted at a
    finer grain than the effect being measured.
    """
    if ground_truth and str(ground_truth).strip():
        prompt = GROUNDED_RUBRIC_TEMPLATE.format(query=query, ground_truth=ground_truth, summary=summary)
    else:
        prompt = RUBRIC_TEMPLATE.format(query=query, summary=summary)

    if votes <= 1:
        return _judge_once(client, prompt, max_retries=max_retries)

    verdicts = [_judge_once(client, prompt, max_retries=max_retries) for _ in range(votes)]
    tally = Counter(v.label for v in verdicts)
    top_count = max(tally.values())
    winners = [label for label, count in tally.items() if count == top_count]
    if len(winners) == 1:
        winning_label = winners[0]
    else:
        winning_label = sorted(winners, key=lambda l: _LABEL_RANK[l])[len(winners) // 2]
    # Return the rationale that actually belongs to the winning label.
    return next(v for v in verdicts if v.label == winning_label)


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
    parser.add_argument(
        "--votes",
        type=int,
        default=1,
        help="Judge each pair this many times and take the majority label (default 1).",
    )
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required")
    client = genai.Client(api_key=GEMINI_API_KEY)

    rows = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "index": i,
                "query": r.get("query", ""),
                "summary": r.get("summary") or "",
                "ground_truth": r.get("ground_truth") or "",
            })
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
            verdict = judge_pair(
                client,
                row["query"],
                row["summary"],
                ground_truth=row.get("ground_truth"),
                votes=args.votes,
            )
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
