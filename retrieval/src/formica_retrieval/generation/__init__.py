"""Stage 7: turn retrieved rows into prose.

Row formatting (kg_hop_path) is the important half and runs with no LLM: it is
the pipeline's actual retrieval output, and what the deterministic recall metric
measures. The Gemini summary is a reading of that text, not a separate lookup.
"""
