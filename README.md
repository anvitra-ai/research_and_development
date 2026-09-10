# Research & Development

Experimental work for **Avintra Search AI / Graphiti** — knowledge-graph retrieval, template-based Cypher generation, and LLM-powered graph agents over Indian thematic equity data.

## Layout

| Folder | Purpose |
|--------|---------|
| [`retrieval/`](retrieval/) | Formica KG query pipeline: entity resolution, template classification, Cypher execution |
| [`data/`](data/) | Datasets, templates, aliases, pipeline outputs, and evaluation results |
| [`models/`](models/) | Trained classifiers (SVM + DeBERTa checkpoints) |
| [`kg/`](kg/) | India thematic stock knowledge graph (Cypher, docs, explorer) |
| [`graphiti/`](graphiti/) | Graphiti + Neo4j integration, notebooks, and Pydantic AI agent |

## Quick start

1. Install dependencies for the area you are working in (each subfolder has its own `requirements.txt`).
2. Start Neo4j locally and set connection env vars (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`).
3. For the Formica pipeline, run from `retrieval/`:

```bash
cd retrieval
pip install -r requirements.txt
python run_batch_pipeline.py --no-summarize
```

4. For the Graphiti agent, set `GEMINI_API_KEY` and run `graphiti/agent.py`.

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `NEO4J_URI` | retrieval | Bolt URI (default `bolt://localhost:7687`) |
| `NEO4J_USER` | retrieval | Neo4j username |
| `NEO4J_PASSWORD` | retrieval | Neo4j password |
| `GEMINI_API_KEY` | graphiti, retrieval | Google Gemini API key |
| `MODEL_CHOICE` | graphiti | Gemini model name (default `gemini-3.6-flash`) |

## Pipeline overview

Natural-language questions flow through entity linking → template classification → slot filling → Cypher → optional Gemini summarization. See [`retrieval/README.md`](retrieval/README.md) for the full step-by-step flow.
