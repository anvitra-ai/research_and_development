"""Shared path constants for the formica_retrieval package.

Layout:
  research_and_development/
    retrieval/
      src/formica_retrieval/   <- Python package (this file)
      scripts/                 <- CLI entry points (batch run, verification, training)
      legacy/                  <- superseded code kept for reference, not imported
      data/                    <- this package's own datasets/templates/aliases + default outputs
    data/                      <- other subprojects' shared data (ingestion, historical result
                                  archives); only used as a fallback, see DATA_DIR below
    models/                    <- trained classifiers
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RETRIEVAL_ROOT = PACKAGE_ROOT.parent.parent
RD_ROOT = RETRIEVAL_ROOT.parent
REPO_ROOT = RD_ROOT.parent

# The package is self-contained: its own datasets/templates/aliases (and, by
# default, its batch/judge output files) live in retrieval/data/. Older data
# that hasn't been migrated in from research_and_development/data/ -- or a repo
# laid out before this package existed -- still resolves via the fallbacks.
DATA_DIR = (
    RETRIEVAL_ROOT / "data" if (RETRIEVAL_ROOT / "data").exists()
    else RD_ROOT / "data" if (RD_ROOT / "data").exists()
    else REPO_ROOT / "data"
)
MODELS_DIR = RD_ROOT / "models" if (RD_ROOT / "models").exists() else REPO_ROOT / "models"
