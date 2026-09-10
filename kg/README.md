# Knowledge Graph (India Thematic Stocks)

Source material for the **India thematic stock knowledge graph** — causal exposure mapping for NSE/BSE-listed equities across macro themes (e.g. Strait of Hormuz disruption, India AI infrastructure buildout).

> **Not investment advice.** This is a research scaffold mapping transmission channels, not price views or valuations.

## Contents

| File | Description |
|------|-------------|
| `india_theme_kg.md` | Full graph specification: ontology, nodes, typed edges, theme definitions |
| `india_theme_kg.cypher` | Cypher statements to load the graph into Neo4j |
| `india_theme_graph_explorer.html` | Static HTML explorer for browsing the graph |
| `requirements.txt` | Python dependencies for notebooks and tooling |

## Graph ontology (summary)

**Node types:** `SECTOR`, `SUBSECTOR`, `COMPANY`, `PRIVATE_CO`, `COMMODITY`, `GEO`, `POLICY`, `MACRO`, `EVENT`, etc.

**Edge types:** Directed, typed relationships expressing causal exposure (e.g. supply chain, demand, policy transmission).

**Order (1st / 2nd / 3rd):** Hop distance from a shock node — how directly a company is affected.

See `india_theme_kg.md` §1 for the complete ontology and §0 for how to read node/edge notation.

## Loading into Neo4j

1. Start Neo4j and create a database.
2. Run the statements in `india_theme_kg.cypher` (via Neo4j Browser, `cypher-shell`, or a script).
3. Open `india_theme_graph_explorer.html` in a browser to explore locally, or query via the retrieval pipeline in `../retrieval/`.

## Related

- Entity aliases used at query time: `../data/kg_entity_aliases.json`
- Query templates: `../data/formica_query_templates.json`
