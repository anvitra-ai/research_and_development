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

## Modules

| Module | Role |
|--------|------|
| `formica_pipeline.py` | End-to-end pipeline orchestration |
| `run_batch_pipeline.py` | CLI for batch processing over a CSV |
| `entity_resolver.py` | Entity linking (gazetteer, aliases, NER, embeddings) |
| `formica_template_classifier.py` | SVM template classifier + rule-based fallback |
| `formica_template_resolver.py` | Slot extraction, Cypher expansion, execution |
| `formica_template_classes.py` | Template class constants |
| `kg_template_resolver.py` | Legacy KG template resolver |
| `generate_formica_training_data.py` | Build labeled training data |
| `generate_query_template_dataset.py` | Synthetic query dataset generator |
| `paths.py` | Shared path constants (`DATA_DIR`, `MODELS_DIR`) |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if not already installed
```

Ensure Neo4j is running and set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. For summarization, set `GEMINI_API_KEY`.

## Usage

Batch run over training queries (no LLM summarization):

```bash
python run_batch_pipeline.py --no-summarize
```

Resume an interrupted run:

```bash
python run_batch_pipeline.py --no-summarize --resume
```

Custom input/output:

```bash
python run_batch_pipeline.py \
  --input ../data/formica_template_test.csv \
  --output-csv ../data/my_results.csv \
  --output-jsonl ../data/my_results.jsonl \
  --no-summarize
```

## Models & data

- Classifier: `../models/formica-template-classifier.joblib`
- Templates: `../data/formica_query_templates.json`
- Aliases: `../data/kg_entity_aliases.json`
- Node/edge gazetteers: `data_node_names.csv`, `data_edge_names.csv` (in this folder)

## Notebooks

- `spacy_subject_extraction.ipynb` — exploratory subject extraction with spaCy
