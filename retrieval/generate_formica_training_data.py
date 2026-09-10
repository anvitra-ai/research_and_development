"""Bootstrap Formica 21-class training data and train the SVM classifier."""

import argparse
import json
from pathlib import Path

import pandas as pd

from formica_template_classifier import (
    FormicaTemplateClassifier,
    rule_label_formica,
    save_labels,
)
from paths import DATA_DIR, MODELS_DIR

TRAIN_CSV = DATA_DIR / "deberta_stock_impact_train.csv"
TEST_CSV = DATA_DIR / "deberta_stock_impact_test.csv"
OUTPUT_TRAIN = DATA_DIR / "formica_template_train.csv"
OUTPUT_TEST = DATA_DIR / "formica_template_test.csv"
MODEL_PATH = MODELS_DIR / "formica-template-classifier.joblib"


def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["formica_label"] = [
        rule_label_formica(str(text), str(domain) if pd.notna(domain) else None)
        for text, domain in zip(out["text"], out.get("label", [None] * len(out)))
    ]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default=str(TRAIN_CSV))
    parser.add_argument("--test-input", default=str(TEST_CSV))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_input)
    train_df = label_dataframe(train_df)
    train_df.to_csv(OUTPUT_TRAIN, index=False)

    if Path(args.test_input).exists():
        test_df = pd.read_csv(args.test_input)
        test_df = label_dataframe(test_df)
        test_df.to_csv(OUTPUT_TEST, index=False)

    save_labels()

    clf = FormicaTemplateClassifier(model_path=args.model_path)
    clf.fit(train_df["text"].astype(str).tolist(), train_df["formica_label"].tolist())
    clf.save()

    dist = train_df["formica_label"].value_counts().to_dict()
    print(f"Saved training data to {OUTPUT_TRAIN}")
    print(f"Saved classifier to {args.model_path}")
    print("Label distribution:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label:<22} {count}")

    if Path(args.test_input).exists():
        test_df = pd.read_csv(OUTPUT_TEST)
        preds = [clf.predict(t)[0] for t in test_df["text"].astype(str)]
        acc = sum(p == g for p, g in zip(preds, test_df["formica_label"])) / len(test_df)
        print(f"\nRule-label agreement on test (bootstrap): {acc:.3f}")


if __name__ == "__main__":
    main()
