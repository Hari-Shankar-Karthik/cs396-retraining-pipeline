import mlflow
import mlflow.transformers
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Dummy training logic - just to get a model into MLflow
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        input_example="Example string",
    )

    print("✅ Logged model to MLflow!")
    print("🔑 Run ID:", run.info.run_id)
