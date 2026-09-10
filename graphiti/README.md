# Graphiti

Integration with **[Graphiti](https://github.com/getzep/graphiti)** — temporal knowledge graphs backed by Neo4j, with Gemini for embeddings, reranking, and agent tooling.

## Contents

| File | Description |
|------|-------------|
| `agent.py` | Pydantic AI agent with a `search_graphiti` tool over a Graphiti client |
| `graphiti_neo4j.ipynb` | Notebook: build and query a Graphiti graph in Neo4j |
| `faiss_neo4j_graph_embeddings.ipynb` | Notebook: FAISS + Neo4j graph embeddings experiments |
| `data.csv` | Sample node data for ingestion |
| `data_edges.csv` | Sample edge data for ingestion |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in this directory (or project root):

```env
GEMINI_API_KEY=your_key_here
MODEL_CHOICE=gemini-3.6-flash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Running the agent

```bash
python agent.py
```

The agent uses Gemini via Pydantic AI and exposes an async `search_graphiti` tool that queries the Graphiti knowledge graph for temporal facts.

## Notebooks

- **`graphiti_neo4j.ipynb`** — end-to-end Graphiti + Neo4j workflow
- **`faiss_neo4j_graph_embeddings.ipynb`** — hybrid retrieval with FAISS vector search over graph embeddings

## Related

- Formica Cypher pipeline (template-based, non-Graphiti): `../retrieval/`
- Static India thematic KG: `../kg/`
