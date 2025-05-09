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
import pandas as pd
from mlflow.models.signature import infer_signature

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, local_files_only=True
        )
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

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
        save_strategy="epoch",
        report_to=["none"],
        disable_tqdm=True,
        fp16=torch.cuda.is_available(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("code-grader")
        with mlflow.start_run():
            # Convert data to DataFrame for logging
            dataset_df = pd.DataFrame(data, columns=["code", "criteria", "label"])
            mlflow.log_input(
                mlflow.data.from_pandas(dataset_df, source="grading_data.db"),
                context="training",
            )

            # Log hyperparameters
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "num_train_epochs": training_args.num_train_epochs,
                    "batch_size": training_args.per_device_train_batch_size,
                    "fp16": training_args.fp16,
                }
            )

            # Train the model
            train_result = trainer.train()
            mlflow.log_metric("training_loss", train_result.training_loss)

            # Define model signature
            sample_input = {
                "text": ["Grade this code: sample code | Criteria: sample criteria"]
            }
            sample_output = {"label": [0]}
            signature = infer_signature(sample_input, sample_output)

            # Log and register the model
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="code_grader_model",
                registered_model_name="CodeGraderModel",
                signature=signature,
                metadata={"task": "code_grading", "dataset_size": len(data)},
            )
            logger.info(
                "Model retrained, logged, and registered to MLflow Model Registry"
            )
            with open("/tmp/new_model_ready", "w") as f:
                f.write("ready")
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise


if __name__ == "__main__":
    retrain_model()
