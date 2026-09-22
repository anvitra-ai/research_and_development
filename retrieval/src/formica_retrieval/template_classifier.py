"""Formica et al. (2023) template classifier: PoS + discriminative keywords + SVM.

Takes a *masked* query (entities replaced with <TYPE> tags) and predicts which
Formica template class best fits the question shape, e.g. F_Simple for "which
companies …", F_QuantCount for "how much … transits …".

Features: bag-of-words bi-grams + PoS tag sequence + discriminative keyword hits.
Model: LinearSVC trained on formica_template_train.csv.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .paths import DATA_DIR, MODELS_DIR
from .template_classes import (
    FORMICA_TEMPLATE_CLASSES,
    DISCRIMINATIVE_KEYWORDS,
    BANKING_SEGMENT_KEYWORDS,
)

DEFAULT_MODEL_PATH = MODELS_DIR / "formica-template-classifier.joblib"
DEFAULT_LABELS_PATH = DATA_DIR / "formica_template_labels.json"

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP


def extract_pos_sequence(text: str) -> str:
    """Syntactic feature: space-separated PoS tags (Formica step 1)."""
    nlp = _get_nlp()
    if not nlp:
        return ""
    doc = nlp(text)
    return " ".join(tok.pos_ for tok in doc)


def extract_semantic_keywords(text: str) -> str:
    """Semantic feature: discriminative keyword hits from paper Table 1."""
    q = text.lower()
    hits: list[str] = []
    for label, keywords in DISCRIMINATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                hits.append(f"{label}:{kw.replace(' ', '_')}")
    return " ".join(hits)


def build_feature_text(text: str) -> str:
    """Combine raw question, PoS tags, and keyword features."""
    pos = extract_pos_sequence(text)
    sem = extract_semantic_keywords(text)
    return f"{text} POS_{pos} KW_{sem}"


def _identity_transform(x):
    return x


def build_pipeline() -> Pipeline:
    """CountVectorizer (bi-grams) + LinearSVC as in Formica et al."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "bow",
                            CountVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                lowercase=True,
                                min_df=1,
                            ),
                        ),
                        (
                            "pos_kw",
                            CountVectorizer(
                                analyzer="word",
                                token_pattern=r"\S+",
                                ngram_range=(1, 2),
                                lowercase=False,
                                min_df=1,
                            ),
                        ),
                    ],
                    transformer_weights={"bow": 1.0, "pos_kw": 1.5},
                ),
            ),
            ("clf", LinearSVC()),
        ]
    )


def rule_label_formica(query: str, domain_label: str | None = None) -> str:
    """Bootstrap labels using discriminative keywords + domain hints."""
    q = query.lower()
    # "most recently reported quarter" etc. is a standard qualifier phrase in this
    # dataset (when the fact is from), not a request for a maximum/superlative --
    # strip it before matching bare "most" so it doesn't false-trigger F_QuantMax.
    q = q.replace("most recently", " ").replace("most recent", " ")

    def has(*patterns: str) -> bool:
        return any(p in q for p in patterns)

    if has(" or ", " either ") and not has("more than", "less than"):
        return "F_LogUnion"
    if has(" and ", " both ") and domain_label in {"T_EventScenario", "T_SectorExposure"}:
        return "F_LogIntersection"
    if has("but not", "without", "except", "excluding"):
        return "F_LogDifference"

    # Cross-company / group-aggregation family -- checked BEFORE the generic
    # "compare"/"highest"/"lowest" keyword checks below, since all three of
    # these shapes contain those same words but need fundamentally different
    # Cypher (rank/aggregate across the whole company universe or a
    # BankingSegment group, not one company or a fixed pair).
    segments_mentioned = [
        seg for seg, kws in BANKING_SEGMENT_KEYWORDS.items() if any(kw in q for kw in kws)
    ]
    # "X's ratio, and how does it compare with the average for its segment?" --
    # one specific company (not named here, but this is text-only labeling)
    # against ITS OWN segment's average. Must come before the plain "compare"
    # check, which would otherwise route this into the two-named-entity F_CompMore.
    if has("average for its segment", "segment average", "peer average", "average for its peer"):
        return "F_CompToGroupAverage"
    # "How do/does Small Finance Banks as a group compare to Private Sector Banks
    # on X?" -- two distinct segments named, no single company at all.
    if len(segments_mentioned) >= 2 or has("as a group compare", "compare... as a group"):
        return "F_GroupAggregate"
    # "Which segment has the highest average X?" -- either a named segment
    # keyword or the bare generic "segment" ("which segment has...", not naming
    # Public/Private/Small Finance specifically), plus "average" plus a
    # superlative; ranks segments themselves, not companies.
    if (segments_mentioned or "segment" in q) and has("average") and has(
        "highest", "lowest", "max", "min", "maximum", "minimum", "largest", "smallest"
    ):
        return "F_GroupAggregate"
    # "Across all 39 listed Indian banks, which reports the highest/lowest X?" or
    # "Which Public Sector Bank has the highest X?" -- ranks ONE company across
    # the whole universe (optionally segment-filtered), no "average" involved.
    if (has("across all", "all 39", "all listed", "all banks") or segments_mentioned) and has(
        "which bank", "which company", "which public sector bank", "which private sector bank",
        "which small finance bank", "report the highest", "report the lowest",
        "reports the highest", "reports the lowest",
    ):
        return "F_GlobalRank"

    # Check comparison intent BEFORE quantity keywords: "Compare X and Y's gross
    # NPA... by how much?" contains "how much" too, but two-entity comparison is
    # the far more specific/decisive signal here -- checking quantity keywords
    # first misroutes the large majority of comparison questions in this dataset
    # (77% of what would otherwise be tagged F_QuantCount) into the single-entity
    # count template, which can't answer a two-company comparison at all.
    if domain_label == "T_CompareExposure" or has("compare", "peer", " vs ", " versus "):
        if has("less", "lower", "fewer"):
            return "F_CompLess"
        if has("approximately", "around", "similar"):
            return "F_CompApprox"
        return "F_CompMore"

    if has("how many", "how much", "number of", "count of"):
        if has("more than", "greater than", "higher than"):
            return "F_CompCountMore"
        if has("less than", "fewer than", "lower than"):
            return "F_CompCountLess"
        if has("approximately", "around", "about"):
            return "F_CompCountApprox"
        if has("at most", "atmost", "no more than"):
            return "F_QuantCountAtmost"
        if has("at least", "atleast"):
            return "F_QuantCountAtleast"
        if has("exactly", "equal to", "precisely"):
            return "F_QuantCountEqual"
        if has("approximately", "around"):
            return "F_QuantCountApprox"
        return "F_QuantCount"

    if has("more than", "greater", "higher", "larger"):
        return "F_CompMore"
    if has("less than", "lower", "fewer", "smaller"):
        return "F_CompLess"
    if has("approximately", "around", "about the same", "similar"):
        return "F_CompApprox"

    if has("at most", "atmost", "up to", "maximum"):
        return "F_QuantAtmost"
    if has("at least", "atleast", "minimum"):
        return "F_QuantAtleast"
    if has("exactly", "equal", "precisely"):
        return "F_QuantEqual"
    # "most"/"least" deliberately excluded: bare "most" collides with far too many
    # non-superlative senses in this dataset ("most sensitive to risk", "prioritize
    # most", "most of the figures", ...) -- none of the 46 such queries checked are
    # genuine "give me the maximum value" questions this template can answer.
    if has("max", "maximum", "highest", "largest"):
        return "F_QuantMax"
    if has("min", "minimum", "lowest", "smallest"):
        return "F_QuantMin"
    if has("approximately", "around", "roughly"):
        return "F_QuantApprox"

    if domain_label in {
        "T_TransitRisk",
        "T_SectorExposure",
        "T_Beneficiary",
        "T_SupplyDisruption",
    } and has("how much", "how many", "share", "what share"):
        return "F_QuantCount"

    return "F_Simple"


class FormicaTemplateClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = list(FORMICA_TEMPLATE_CLASSES)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        X = [build_feature_text(t) for t in texts]
        self.classes_ = sorted(set(labels))
        self.pipeline = build_pipeline()
        self.pipeline.fit(X, labels)

    def predict(self, text: str) -> tuple[str, float]:
        hits = self.predict_proba_like(text, top_k=1)
        return hits[0][0], hits[0][1]

    def predict_proba_like(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        if self.pipeline is None:
            self.load()
        X = [build_feature_text(text)]
        clf: LinearSVC = self.pipeline.named_steps["clf"]
        feat = self.pipeline.named_steps["features"].transform(X)
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(feat)[0]
            if np.ndim(scores) == 0:
                scores = np.array([scores, -scores])
            exp = np.exp(scores - scores.max())
            probs = exp / exp.sum()
            pairs = list(zip(clf.classes_, probs))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [(lbl, float(score)) for lbl, score in pairs[:top_k]]
        label = self.pipeline.predict(X)[0]
        return [(label, 0.5)]

    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Classifier not trained")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, self.model_path)

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Formica classifier not found at {self.model_path}. "
                "Run scripts/train_template_classifier.py first."
            )
        payload = joblib.load(self.model_path)
        self.pipeline = payload["pipeline"]
        self.classes_ = payload.get("classes", list(FORMICA_TEMPLATE_CLASSES))


def save_labels(path: str | Path = DEFAULT_LABELS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "classes": FORMICA_TEMPLATE_CLASSES,
                "descriptions": {
                    c: DISCRIMINATIVE_KEYWORDS.get(c, []) for c in FORMICA_TEMPLATE_CLASSES
                },
            },
            f,
            indent=2,
        )


def extract_threshold(query: str) -> int | float | None:
    """Parse numeric bound from comparative/quantitative questions."""
    q = query.lower()
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*%?", q)
    if m:
        val = float(m.group(1))
        return int(val) if val.is_integer() else val
    if "half" in q:
        return 0.5
    return None
