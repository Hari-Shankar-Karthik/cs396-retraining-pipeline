import sqlite3
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import mlflow
import mlflow.transformers


def load_data():
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT code, criteria, label FROM graded_samples")
    rows = cursor.fetchall()
    conn.close()
    return rows


def preprocess(rows):
    inputs = [
        f"Grade this code: {code} | Criteria: {criteria}" for code, criteria, _ in rows
    ]
    labels = [label for _, _, label in rows]
    return inputs, labels


def retrain_model():
    data = load_data()
    if len(data) == 0:
        print("No new data to retrain.")
        return

    inputs, labels = preprocess(data)

    mlflow.set_experiment("code-grader")
    with mlflow.start_run():
        model_name = "distilbert-base-uncased"  # basic placeholder
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

        # quick training — can be replaced with Trainer/accelerate
        for i, input_text in enumerate(inputs):
            print(f"Would train on: {input_text} => {labels[i]}")

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="code_grader_model",
        )

        print("Retraining done and model logged to MLflow.")


if __name__ == "__main__":
    retrain_model()
