"""Train the Formica template classifier on synthetic, genuinely-labelled data.

Replaces generate_formica_training_data.py, which labelled its training set with
rule_label_formica() -- the same function the model was then scored against and
overridden by, making its reported 100% accuracy meaningless (see
synthetic_queries.py for the full explanation).

Here the label comes from the generating frame, so it is independent of any rule
or classifier. The script reproduces the paper's ablation (Formica et al. 2023,
Tables 8 and 9): syntactic features alone, then syntactic + semantic, per-class
precision/recall/F-score, and a held-out test split.

  python3 scripts/train_template_classifier.py
  python3 scripts/train_template_classifier.py --per-class 300 --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from formica_retrieval.paths import DATA_DIR  # noqa: E402
from formica_retrieval.synthetic_queries import class_coverage, generate  # noqa: E402
from formica_retrieval.template_classifier import (  # noqa: E402
    FormicaTemplateClassifier,
    build_pipeline,
    extract_pos_sequence,
    extract_semantic_keywords,
    save_labels,
)

TRAIN_OUT = DATA_DIR / "formica_synthetic_train.csv"
TEST_OUT = DATA_DIR / "formica_synthetic_test.csv"


def _fit_and_score(train_texts, train_labels, test_texts, test_labels, featurizer, name):
    pipe = build_pipeline()
    pipe.fit([featurizer(t) for t in train_texts], train_labels)
    preds = pipe.predict([featurizer(t) for t in test_texts])
    acc = sum(p == g for p, g in zip(preds, test_labels)) / len(test_labels)
    print(f"\n{name}: accuracy {acc:.3f}")
    return pipe, preds, acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--split",
        choices=["frames", "random"],
        default="frames",
        help="frames = train/test share no phrasing (honest); random = shares frames (inflated).",
    )
    parser.add_argument("--no-save", action="store_true", help="Evaluate only; don't write the model.")
    parser.add_argument("--per-class-report", action="store_true", help="Print paper Table 9 equivalent.")
    args = parser.parse_args()

    coverage = class_coverage()
    thin = {c: n for c, n in coverage.items() if n < 3}
    print(f"Classes: {len(coverage)}   frames/class: "
          f"min={min(coverage.values())} max={max(coverage.values())}")
    if thin:
        print(f"WARNING: too few frames to vary phrasing for: {thin}")

    pairs = list(generate(per_class=args.per_class))
    texts = [t for t, _ in pairs]
    labels = [l for _, l in pairs]
    print(f"Generated {len(pairs)} examples across {len(set(labels))} classes")

    counts = pd.Series(labels).value_counts()
    if counts.min() < args.per_class:
        print(f"NOTE: some classes under target ({counts.min()} < {args.per_class}) -- "
              "frame set too small to produce that many distinct questions.")

    if args.split == "frames":
        # The split that measures generalisation: train and test share NO frame,
        # so the model must transfer to phrasings it has never seen -- which is
        # the situation every real benchmark question presents.
        tr_pairs = list(generate(per_class=args.per_class, frame_slice="train"))
        te_pairs = list(generate(per_class=args.per_class, frame_slice="holdout", seed=99))
        tr_x, tr_y = [t for t, _ in tr_pairs], [l for _, l in tr_pairs]
        te_x, te_y = [t for t, _ in te_pairs], [l for _, l in te_pairs]
        print(f"FRAME-disjoint split -- train {len(tr_x)} / test {len(te_x)} (no shared phrasings)")
    else:
        # Stratify so every class appears in both halves. NOTE: this split shares
        # frames across train/test and therefore measures frame memorisation, not
        # generalisation. Reported only for comparison with the paper's Table 8.
        tr_x, te_x, tr_y, te_y = train_test_split(
            texts, labels, test_size=args.test_size, random_state=7, stratify=labels
        )
        print(f"RANDOM split -- train {len(tr_x)} / test {len(te_x)} "
              "(shares frames; inflates accuracy)")

    # Paper Table 8: syntactic features alone, then syntactic + semantic.
    _fit_and_score(tr_x, tr_y, te_x, te_y, lambda t: f"POS_{extract_pos_sequence(t)}",
                   "Only syntactic features (PoS)")
    pipe, preds, acc = _fit_and_score(
        tr_x, tr_y, te_x, te_y,
        lambda t: f"{t} POS_{extract_pos_sequence(t)} KW_{extract_semantic_keywords(t)}",
        "Syntactic and semantic features",
    )

    if args.per_class_report:
        print("\nPer-class (paper Table 9 equivalent):")
        print(classification_report(te_y, preds, zero_division=0))

    pd.DataFrame({"text": tr_x, "formica_label": tr_y}).to_csv(TRAIN_OUT, index=False)
    pd.DataFrame({"text": te_x, "formica_label": te_y}).to_csv(TEST_OUT, index=False)
    print(f"\nWrote {TRAIN_OUT.name} / {TEST_OUT.name}")

    if args.no_save:
        print("--no-save: model not written.")
        return

    clf = FormicaTemplateClassifier()
    clf.fit(texts, labels)  # final model uses all the data
    clf.save()
    save_labels()
    print(f"Saved model to {clf.model_path}")


if __name__ == "__main__":
    main()
