import sqlite3
import os
import mlflow
import mlflow.transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import logging
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data():
    try:
        conn = sqlite3.connect("grading_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT code, criteria, label FROM graded_samples")
        rows = cursor.fetchall()
        conn.close()
        logger.info(f"Loaded {len(rows)} samples from database")
        return rows
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess(rows):
    if not rows:
        raise ValueError("No data to preprocess")
    inputs = [
        f"Grade this code: {code} | Criteria: {criteria}" for code, criteria, _ in rows
    ]
    labels = [label for _, _, label in rows]
    return inputs, labels

def retrain_model():
    logger.info("Starting model retraining")
    data = load_data()
    if len(data) < 10:
        logger.warning("Insufficient data for retraining. Need at least 10 samples.")
        return

    inputs, labels = preprocess(data)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_strategy="epoch",
        report_to=["none"],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    try:
        mlflow.set_experiment("code-grader")
        with mlflow.start_run():
            trainer.train()
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="code_grader_model",
                registered_model_name="CodeGraderModel",
            )
            logger.info("Model retrained and logged to MLflow")

            with open("/tmp/new_model_ready", "w") as f:
                f.write("ready")
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise

if __name__ == "__main__":
    retrain_model()
