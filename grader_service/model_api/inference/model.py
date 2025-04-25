from transformers import pipeline
import mlflow.transformers
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model():
    try:
        # Load the latest production model from MLflow
        model_uri = "models:/CodeGraderModel/Production"
        model = mlflow.transformers.load_model(model_uri)
        logger.info("Loaded model from MLflow")
    except Exception as e:
        logger.error(f"Failed to load from MLflow: {e}")
        # Fallback to a default model
        model = pipeline("text-classification", model="distilbert-base-uncased")
        logger.warning("Loaded default distilbert-base-uncased model")
    return model


def grade_submission(code, criteria, model=None):
    if model is None:
        model = load_model()

    input_text = f"Grade this code: {code} | Criteria: {criteria}"
    result = model(input_text)

    label = result[0]["label"]
    score = result[0]["score"]
    return {"label": label, "confidence": score}
