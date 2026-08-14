import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = PROJECT_ROOT / "classifier_corpus.csv"
ARTIFACT_FILE = PROJECT_ROOT / "assets" / "classifier" / "route_classifier.joblib"


def load_training_rows(corpus_file=CORPUS_FILE):
    with Path(corpus_file).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    texts = [row["text"].strip() for row in rows if row.get("text", "").strip()]
    labels = [int(row["label"]) for row in rows if row.get("text", "").strip()]
    return texts, labels


def train_route_classifier(texts, labels):
    classifier = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


def main():
    texts, labels = load_training_rows()
    classifier = train_route_classifier(texts, labels)

    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, ARTIFACT_FILE)
    print(f"Route classifier saved to: {ARTIFACT_FILE}")
    print(f"Training samples: {len(texts)}")
    print(f"Artifact size: {ARTIFACT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
