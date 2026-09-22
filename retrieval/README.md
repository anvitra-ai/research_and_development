# Retrieval

Python package for **Formica KG query retrieval** — turning natural-language questions into Cypher, executing against Neo4j, and optionally summarizing results with Gemini.

## Pipeline flow

1. **Entity resolution (pass 1)** — gazetteer, aliases, NER, semantic search
2. **Template classification** — mask entities as `<TYPE>` tags; predict Formica class (`F_Simple`, `F_QuantCount`, …) via SVM
3. **Entity resolution (pass 2)** — re-link with template-aware type preferences
4. **Entity enrichment** — pull additional nodes required by the template
5. **Template expansion** — map entities to triplet slots and build Cypher
6. **Cypher execution** — run against Neo4j with fallbacks (propagation, transit, beneficiaries, …)
7. **Optional summarization** — format KG rows and summarize with Gemini
8. **Optional LLM-as-judge** — grade the summary for relevance against a rubric (or a labeled ground-truth answer, when available)

## Layout

```
retrieval/
  src/formica_retrieval/   Importable package — the pipeline itself
    config.py                Neo4j/Gemini credentials (env-var backed, single source of truth)
    paths.py                 DATA_DIR / MODELS_DIR / repo-root constants
    entity_resolver.py       Entity linking (gazetteer, aliases, NER, embeddings)
    template_classes.py      Formica template class constants, relation keyword maps
    template_classifier.py   SVM template classifier + rule-based fallback
    template_resolver.py     Slot extraction, Cypher expansion, execution
    resources.py              PipelineResources + load_resources() (NER/embedder/Neo4j/Gemini)
    summarization.py          KG rows -> hop-path text -> Gemini natural-language answer
    results.py                Result-record shape, CSV/JSONL serialization, batch stats
    llm_judge.py              LLM-as-judge relevance rubric (grounded + ungrounded)
    pipeline.py               process_query() orchestration; re-exports the above for callers
  scripts/                  CLI entry points, run directly (not imported by the package)
    run_batch_pipeline.py     Batch-run the pipeline over a CSV of queries
    verify_data_gaps.py       Ground-truth check: does a "not covered" fact really not exist?
    train_template_classifier.py        Train the SVM template classifier on synthetic labelled data
    analyze_failures.py                 Attribute non-Relevant queries to retrieval vs generation
    check_graph_contract.py             Fail loudly when ingestion drifts from what retrieval expects
    regen_snapshot.py                   Regenerate the characterization-test snapshot
  legacy/                   Superseded code, kept for reference only — not on the import path
  data/                     This package's own datasets/templates/aliases + default run outputs
  api.py                    FastAPI service wrapping the same pipeline for ad-hoc search
  requirements.txt
  .env.example
```

The package is self-contained: `paths.DATA_DIR` resolves to `retrieval/data/` whenever it
exists, so the pipeline never has to reach outside its own directory tree for its
required inputs (templates, aliases, classifier labels, the labeled query set) or
for where its own batch/judge runs write by default. Older, unmigrated data (or a
repo checked out before this package existed) falls back to
`research_and_development/data/`, then the repo root's `data/`.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if not already installed
cp .env.example .env                       # then fill in GEMINI_API_KEY (and NEO4J_* if not local)
```

Neo4j must be running. All credentials are read once, centrally, by `src/formica_retrieval/config.py`.

## Usage

Batch run over the labeled training queries (no LLM summarization):

```bash
cd scripts
python run_batch_pipeline.py --no-summarize
```

Resume an interrupted run:

```bash
python run_batch_pipeline.py --no-summarize --resume
```

Custom input/output, with summarization and LLM-as-judge (historical result
archives still live in `research_and_development/data/` since they were never
moved):

```bash
python run_batch_pipeline.py \
  --input ../../data/formica_template_test.csv \
  --output-csv ../data/my_results.csv \
  --output-jsonl ../data/my_results.jsonl \
  --judge
```

Run the API:

```bash
cd retrieval
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

Verify whether a "not covered" answer reflects a real gap in the graph or a retrieval bug:

```bash
cd scripts
python verify_data_gaps.py \
  --input ../data/my_results.csv \
  --keep-output ../data/no_gap.csv \
  --gap-output ../data/gap.csv \
  --report-output ../data/gap_report.csv
```

## Models & data

- Classifier: `../models/formica-template-classifier.joblib`
- Templates: `data/formica_query_templates.json`
- Aliases: `data/kg_entity_aliases.json`
- Classifier labels: `data/formica_template_labels.json`
- Labeled query set (993 banking queries + ground truth): `data/banking_queries/indian_banks_993_queries.csv`
- Synthetic classifier training/test split (written by `train_template_classifier.py`):
  `data/formica_synthetic_{train,test}.csv`
- `data/deberta_stock_impact_{train,test}.csv` were deleted along with the
  `generate_formica_training_data.py` bootstrap trainer that consumed them. Only
  `legacy/spacy_subject_extraction.ipynb` still refers to them; regenerate with
  `legacy/generate_query_template_dataset.py` if that notebook is ever needed.

## Legacy

`legacy/` holds code superseded by the current package (a pre-Formica KG template resolver, an older synthetic-dataset generator, an exploratory spaCy notebook, and two unused gazetteer CSVs). None of it is imported by the pipeline, API, or scripts — kept only as historical reference.
