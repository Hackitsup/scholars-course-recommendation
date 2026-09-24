from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "aiml_training_data.csv",
    Path(r"C:\Users\supre\Downloads\aiml_training_data.csv"),
]
MODEL_PATH = ROOT / "artifacts" / "intent_classifier.joblib"


def find_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Place aiml_training_data.csv in data/ or Downloads.")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_and_save() -> dict[str, object]:
    data = pd.read_csv(find_data_path())
    required_columns = {"text", "intent"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = data.dropna(subset=["text", "intent"])
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["intent"],
        test_size=0.2,
        random_state=42,
        stratify=data["intent"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(data["intent"].unique())

    metrics = {
        "rows": len(data),
        "classes": labels,
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return metrics


if __name__ == "__main__":
    result = train_and_save()
    print(f"Rows: {result['rows']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(result["classification_report"])
    print(f"Saved model to {MODEL_PATH}")
