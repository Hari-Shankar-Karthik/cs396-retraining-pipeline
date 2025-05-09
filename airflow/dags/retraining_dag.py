from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def retrain_model():
    try:
        result = subprocess.run(
            ["python", "/raid/ganesh/saurav/hari/grader_service/retrain.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Retraining completed: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e.stderr}")
        raise


def deploy_model():
    try:
        # Check if new model is ready
        if os.path.exists("/tmp/new_model_ready"):
            result = subprocess.run(
                ["python", "/raid/ganesh/saurav/hari/grader_service/deploy_model.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Model deployed: {result.stdout}")
            os.remove("/tmp/new_model_ready")  # Clean up signal file
        else:
            logger.info("No new model to deploy")
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e.stderr}")
        raise


default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_code_grader",
    default_args=default_args,
    description="Retrains and deploys the code grading model weekly",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    retrain_task = PythonOperator(
        task_id="retrain_task",
        python_callable=retrain_model,
    )
    deploy_task = PythonOperator(
        task_id="deploy_task",
        python_callable=deploy_model,
    )

    retrain_task >> deploy_task  # Chain tasks
