"""CLI entry point for batch-running the Formica KG query pipeline.

Reads queries from a CSV, runs formica_pipeline.process_query() on each row,
and appends results to CSV + JSONL. Use --resume to skip already-processed queries.

Run from anywhere:
  python research_and_development/retrieval/run_batch_pipeline.py --no-summarize
  cd research_and_development/retrieval && python run_batch_pipeline.py --no-summarize
"""

import argparse
import sys
from pathlib import Path

_RETRIEVAL_ROOT = Path(__file__).resolve().parent
if str(_RETRIEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_ROOT))

import pandas as pd
from tqdm import tqdm

from formica_pipeline import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_OUTPUT_JSONL,
    LEGACY_TRAIN_CSV,
    TRAIN_CSV,
    append_result,
    gold_label_from_row,
    load_resources,
    print_summary_stats,
    process_query,
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
    parser.add_argument("--resume", action="store_true", help="Skip queries already in the output CSV.")
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
    existing = pd.read_csv(csv_path)
    done = set(existing["query"].astype(str))
    print(f"Resuming: {len(done)} queries already processed")
    return done


def main() -> None:
    args = parse_args()
    input_df = load_input_frame(args.input, args.offset, args.limit)
    csv_path = Path(args.output_csv)
    jsonl_path = Path(args.output_jsonl)
    done_queries = load_done_queries(csv_path, args.resume)

    # Load models, Neo4j connection, and node embeddings once for the whole batch.
    resources = load_resources(summarize=not args.no_summarize)
    write_header = not csv_path.exists() or not args.resume

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Pipeline"):
        query = str(row["text"])
        if query in done_queries:
            continue
        result = process_query(
            query,
            gold_label_from_row(row),
            resources,
            summarize=not args.no_summarize,
        )
        append_result(result, csv_path, jsonl_path, write_header=write_header)
        write_header = False

    print_summary_stats(csv_path)


if __name__ == "__main__":
    main()
