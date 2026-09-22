"""Shared path constants for the retrieval package.

Layout:
  research_and_development/
    retrieval/   <- Python modules (this file)
    data/        <- datasets, templates, aliases, pipeline outputs
    models/      <- trained classifiers
"""

from __future__ import annotations

from pathlib import Path

RETRIEVAL_ROOT = Path(__file__).resolve().parent
RD_ROOT = RETRIEVAL_ROOT.parent
REPO_ROOT = RD_ROOT.parent

# Prefer research_and_development/{data,models}; fall back to repo root copies.
DATA_DIR = RD_ROOT / "data" if (RD_ROOT / "data").exists() else REPO_ROOT / "data"
MODELS_DIR = RD_ROOT / "models" if (RD_ROOT / "models").exists() else REPO_ROOT / "models"
