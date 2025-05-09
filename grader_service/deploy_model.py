import mlflow
import logging
from model_api.inference.model import load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def deploy_model():
    try:
        client = mlflow.tracking.MlflowClient()
        registered_model_name = "CodeGraderModel"
        latest_versions = client.get_latest_versions(
            registered_model_name, stages=["Production"]
        )

        if not latest_versions:
            logger.warning("No production model found")
            return

        model_uri = f"models:/{registered_model_name}/Production"
        new_model = mlflow.transformers.load_model(model_uri)

        from model_api.views import model, current_model_version

        model.model = new_model["model"]
        model.tokenizer = new_model["tokenizer"]
        current_model_version = latest_versions[0].version  # Update model version

        logger.info(f"Successfully deployed new model version {current_model_version}")
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise


if __name__ == "__main__":
    deploy_model()
