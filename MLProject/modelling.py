from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "phishing_email_detection_2026_dataset_preprocessing.csv"
TRACKING_DIR = BASE_DIR / "mlruns"
ARTIFACT_DIR = BASE_DIR / "artifacts"

TEXT_COLUMNS = ["sender_email_clean", "subject_clean"]
NUMERIC_COLUMNS = [
    "has_link",
    "has_attachment",
    "urgency_score",
    "spelling_errors",
    "email_length_words",
    "subject_word_count",
]
CATEGORICAL_COLUMNS = ["sender_domain"]
TARGET_COLUMN = "is_phishing"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("sender_text", TfidfVectorizer(max_features=3000), "sender_email_clean"),
            ("subject_text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "subject_clean"),
            ("domain", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def save_artifacts(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    with (ARTIFACT_DIR / "classification_report.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    confusion = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion).plot(ax=axis, colorbar=False)
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=roc_axis)
    roc_axis.set_title("ROC Curve")
    roc_figure.tight_layout()
    roc_figure.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=150)
    plt.close(roc_figure)

    return metrics


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("workflow_ci_basic")
    mlflow.sklearn.autolog(log_input_examples=True, silent=True)

    df = load_dataset()
    features = df[TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    target = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline()

    model.fit(x_train, y_train)
    metrics = save_artifacts(model, x_test, y_test)
    mlflow.log_metrics(metrics)
    mlflow.log_param("dataset_path", DATA_PATH.name)
    mlflow.log_param("train_rows", int(x_train.shape[0]))
    mlflow.log_param("test_rows", int(x_test.shape[0]))
    mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="artifacts")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Workflow CI training selesai.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
