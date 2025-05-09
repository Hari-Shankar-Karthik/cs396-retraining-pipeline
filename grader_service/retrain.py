import sqlite3
import os
import mlflow
from mlflow.tracking import MlflowClient
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
from datetime import datetime
import uuid
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def ensure_retrain_status_schema():
    """Ensure the retrain_status table has the correct schema."""
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    logger.info(f"Connecting to database at: {os.path.abspath('grading_data.db')}")

    # Create table if it doesn't exist
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS retrain_status
           (run_id TEXT PRIMARY KEY, status TEXT, error TEXT, start_time TEXT, model_version TEXT)"""
    )

    # Check if model_version column exists
    cursor.execute("PRAGMA table_info(retrain_status)")
    columns = [info[1] for info in cursor.fetchall()]
    if "model_version" not in columns:
        logger.info("Adding model_version column to retrain_status table")
        cursor.execute("ALTER TABLE retrain_status ADD COLUMN model_version TEXT")

    conn.commit()
    conn.close()


def update_retrain_status(run_id, status, model_version=None, error=None):
    """Update retraining status and model version in database."""
    ensure_retrain_status_schema()
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE retrain_status SET status = ?, error = ?, model_version = ? WHERE run_id = ?",
        (status, error, model_version, run_id),
    )
    conn.commit()
    conn.close()


def load_data():
    """Load graded samples from SQLite database."""
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
    """Preprocess data by combining code and criteria, extracting labels."""
    if not rows:
        raise ValueError("No data to preprocess")
    inputs = [
        f"Grade this code: {code} | Criteria: {criteria}" for code, criteria, _ in rows
    ]
    labels = [label for _, _, label in rows]
    return inputs, labels


def transition_to_production(model_version):
    client = MlflowClient()
    try:
        # Transition the model version to the Production stage
        client.transition_model_version_stage(
            name="CodeGraderModel", version=model_version, stage="Production"
        )
        print(f"Model version {model_version} transitioned to Production.")
    except Exception as e:
        print(f"Failed to transition model to production: {e}")


def retrain_model(run_id=None):
    """Retrain the code grading model and log to MLflow."""
    logger.info("Starting model retraining")

    # Ensure database schema is correct
    ensure_retrain_status_schema()

    # Load and validate data
    data = load_data()
    if len(data) < 10:
        logger.warning("Insufficient data for retraining. Need at least 10 samples.")
        if run_id:
            update_retrain_status(run_id, "failed", error="Insufficient training data")
        return

    # Preprocess data
    inputs, labels = preprocess(data)

    # Initialize model and tokenizer
    model_name = "distilbert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, local_files_only=True
        )
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        raise

    # Tokenize inputs
    encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    )

    # Define training arguments
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

    # Move model to appropriate device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    try:
        # Configure MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_registry_uri("http://localhost:5000")
        mlflow.set_experiment("code-grader")

        # Generate descriptive run name with timestamp
        run_name = f"code_grader_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        with mlflow.start_run(run_name=run_name):
            # Log dataset
            dataset_df = pd.DataFrame(data, columns=["code", "criteria", "label"])
            try:
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        dataset_df,
                        source="grading_data.db",
                        name="graded_samples",
                        targets="label",
                    ),
                    context="training",
                )
            except Exception as e:
                logger.warning(f"mlflow.log_input failed: {e}")

            # Log model parameters
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "num_train_epochs": training_args.num_train_epochs,
                    "batch_size": training_args.per_device_train_batch_size,
                    "fp16": training_args.fp16,
                    "device": str(device),
                    "dataset_size": len(dataset),
                }
            )

            # Train model
            train_result = trainer.train()

            # Log training metrics
            metrics = {
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
            }
            mlflow.log_metrics(metrics)

            # Log and register model
            model_info = mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="code_grader_model",
                registered_model_name="CodeGraderModel",
                task="text-classification",
            )

            # Get model version
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions("CodeGraderModel")[-1].version

            # Transition model to Production
            client.transition_model_version_stage(
                name="CodeGraderModel", version=latest_version, stage="Production"
            )

            logger.info(
                "Model retrained, logged, registered, and transitioned to Production"
            )

            # Update retrain status with model version
            if run_id:
                update_retrain_status(run_id, "completed", model_version=latest_version)

            model_version = mlflow.register_model("model", "CodeGraderModel").version
            transition_to_production(model_version)
            update_retrain_status(run_id, "completed", model_version=model_version)

        transition_to_production(model_version)

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        if run_id:
            update_retrain_status(run_id, "failed", error=str(e))
        raise


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    retrain_model(run_id)
