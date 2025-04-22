from transformers import pipeline
import mlflow.transformers
import os


def load_model():
    try:
        MLFLOW_RUN_ID = "45020b161c4a4494ba22e1b7dc60fcc8"
        model_uri = f"runs:/{MLFLOW_RUN_ID}/code_grader_model"
        model = mlflow.transformers.load_model(model_uri)
        print("✅ Loaded model from MLflow")
    except Exception as e:
        print(f"⚠️ Failed to load from MLflow: {e}")
        model = pipeline("text-classification", model="distilbert-base-uncased")
    return model


def grade_submission(code, criteria, model=None):
    if model is None:
        model = load_model()

    input_text = f"Grade this code: {code} | Criteria: {criteria}"
    result = model(input_text)

    label = result[0]["label"]
    score = result[0]["score"]
    return {"label": label, "confidence": score}
