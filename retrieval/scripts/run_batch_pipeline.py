"""CLI entry point for batch-running the Formica KG query pipeline.

Reads queries from a CSV, runs pipeline.process_query() on each row, and
appends results to CSV + JSONL. Use --resume to skip already-processed queries.

Run from anywhere:
  python research_and_development/retrieval/scripts/run_batch_pipeline.py --no-summarize
  cd research_and_development/retrieval/scripts && python run_batch_pipeline.py --no-summarize
"""

import argparse
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from tqdm import tqdm

from formica_retrieval.pipeline import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_JSONL,
    LEGACY_TRAIN_CSV,
    TRAIN_CSV,
    append_result,
    gold_label_from_row,
    load_resources,
    print_summary_stats,
    process_query,
    _ensure_fact_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Formica KG pipeline over a CSV of queries.")
    parser.add_argument(
        "--input",
        default=str(TRAIN_CSV if TRAIN_CSV.exists() else LEGACY_TRAIN_CSV),
        help="Input CSV with a 'text' column (and optional formica_label).",
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-summarize", action="store_true", help="Skip Gemini summarization.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "LLM-as-judge each query/summary pair (Relevant/Somewhat relevant/Not "
            "relevant + rationale), adding judge_label/judge_rationale columns to "
            "the same output. Empty summaries are auto-labeled Not relevant "
            "without an extra LLM call, same as llm_judge.py."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Skip queries already in the output CSV.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Concurrent queries (default 8). The pipeline is ~97%% network wait, so "
            "raising this cuts wall-clock almost linearly; lower it if the Gemini API "
            "starts rate-limiting."
        ),
    )
    parser.add_argument(
        "--judge-votes",
        type=int,
        default=1,
        help="Judge each summary this many times and take the majority label (default 1).",
    )
    return parser.parse_args()


def load_input_frame(path: str, offset: int, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if offset:
        df = df.iloc[offset:]
    if limit:
        df = df.head(limit)
    return df


def load_done_queries(csv_path: Path, resume: bool) -> set[str]:
    if not resume or not csv_path.exists():
        return set()
    try:
        existing = pd.read_csv(csv_path)
        done = set(existing["query"].astype(str))
    except (pd.errors.EmptyDataError, KeyError) as exc:
        # A headerless/malformed/empty file (e.g. left over from an interrupted
        # run) shouldn't hard-crash the whole batch -- treat it as "nothing done
        # yet". Remove it too, so main()'s write_header check (based on whether
        # the file exists) doesn't stay fooled into skipping the header on the
        # fresh restart this triggers.
        print(f"Warning: couldn't read existing output for --resume ({exc}); starting fresh")
        csv_path.unlink(missing_ok=True)
        return set()
    print(f"Resuming: {len(done)} queries already processed")
    return done


def main() -> None:
    args = parse_args()
    input_df = load_input_frame(args.input, args.offset, args.limit)
    csv_path = Path(args.output_csv)
    jsonl_path = Path(args.output_jsonl)
    done_queries = load_done_queries(csv_path, args.resume)

    # Load models, Neo4j connection, and node embeddings once for the whole batch.
    resources = load_resources(summarize=not args.no_summarize, judge=args.judge)

    # Build the fact index up front rather than letting the first query trigger
    # it. process_query builds it lazily via a plain "if is None" check, which is
    # a race once several threads run: they would all see None and build it
    # concurrently, duplicating a ~2s embedding pass and briefly publishing a
    # half-built index. Doing it here makes the workers read-only over it.
    _ensure_fact_index(resources)

    pending = [
        (i, row) for i, (_, row) in enumerate(input_df.iterrows())
        if str(row["text"]) not in done_queries
    ]
    if not pending:
        print("Nothing to do -- every query is already in the output.")
        print_summary_stats(csv_path)
        return

    def run_one(item):
        i, row = item
        ground_truth = (
            str(row["ground_truth"])
            if "ground_truth" in row and pd.notna(row["ground_truth"]) else None
        )
        return i, process_query(
            str(row["text"]),
            gold_label_from_row(row),
            resources,
            summarize=not args.no_summarize,
            judge=args.judge,
            judge_votes=args.judge_votes,
            ground_truth=ground_truth,
        )

    # Queries are independent and the work is ~97% waiting on network calls
    # (measured: LLM summarize 8.0s + judge 1.3s per query against 0.25s of
    # local retrieval), so threads -- not processes -- are the right tool and
    # the GIL is irrelevant here.
    write_lock = Lock()
    write_header = not csv_path.exists() or not args.resume
    results: list[tuple[int, dict]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Pipeline"):
            i, result = fut.result()
            results.append((i, result))
            # Written as each finishes rather than in one batch at the end, so an
            # interrupted run keeps its completed work and --resume can pick it up.
            with write_lock:
                append_result(result, csv_path, jsonl_path, write_header=write_header)
                write_header = False

    # Completion order is nondeterministic under concurrency; rewrite both files
    # in input order so reruns are diffable and positional comparisons between
    # benchmark versions stay valid.
    if results and not done_queries:
        _rewrite_in_order(results, csv_path, jsonl_path)

    print_summary_stats(csv_path)


def _rewrite_in_order(results: list[tuple[int, dict]], csv_path: Path, jsonl_path: Path) -> None:
    ordered = [r for _, r in sorted(results, key=lambda pair: pair[0])]
    csv_path.unlink(missing_ok=True)
    jsonl_path.unlink(missing_ok=True)
    header = True
    for result in ordered:
        append_result(result, csv_path, jsonl_path, write_header=header)
        header = False


if __name__ == "__main__":
    main()
