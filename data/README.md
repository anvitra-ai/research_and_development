# Data

Shared datasets, graph configuration, and pipeline outputs for the retrieval stack. Paths are resolved via `retrieval/paths.py` (`DATA_DIR`).

## Configuration & templates

| File | Description |
|------|-------------|
| `formica_query_templates.json` | Formica Cypher templates and slot definitions |
| `kg_query_templates.json` | Legacy KG query templates |
| `kg_entity_aliases.json` | Entity alias → canonical KG node ID mappings |
| `formica_template_labels.json` | Label index for Formica template classes |
| `deberta_stock_impact_labels.json` | Label index for DeBERTa stock-impact classes |

## Training & test splits

| File | Description |
|------|-------------|
| `formica_template_train.csv` | Formica template classification training set |
| `formica_template_test.csv` | Formica template classification test set |
| ~~`deberta_stock_impact_train.csv`~~ | Deleted — consumer script removed; regenerate via `retrieval/legacy/generate_query_template_dataset.py` |
| ~~`deberta_stock_impact_test.csv`~~ | Deleted — consumer script removed; regenerate via `retrieval/legacy/generate_query_template_dataset.py` |

Each CSV typically has a `text` column (natural-language query) and label columns (`formica_label`, etc.).

## Pipeline outputs

| File | Description |
|------|-------------|
| `formica_pipeline_results.csv` / `.jsonl` | Full batch pipeline run results |
| `formica_pipeline_smoke.csv` / `.jsonl` | Small smoke-test run |
| `deberta_stock_impact_pipeline_results.csv` / `.jsonl` | Legacy DeBERTa pipeline results |
| `formica_relevance_judgments.csv` / `.jsonl` | Human or model relevance judgments |
| `relevance_eval_results.csv` / `.jsonl` | Relevance evaluation metrics |

## Logs

| File | Description |
|------|-------------|
| `formica_batch_pipeline_run.log` | Log from latest Formica batch run |
| `batch_pipeline_run.log` | Log from legacy batch run |

## Notes

- Large `.csv` / `.jsonl` files are generated artifacts; regenerate with `retrieval/run_batch_pipeline.py`.
- Do not commit secrets or API keys in this directory.
