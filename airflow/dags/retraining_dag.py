from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess


def retrain_model():
    subprocess.run(["python", "/raid/ganesh/saurav/hari/grader_service/retrain.py"])


default_args = {
    "owner": "you",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="retrain_code_grader",
    default_args=default_args,
    description="Retrains the code grading model weekly",
    schedule_interval="@weekly",  # or "@daily"
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

task = PythonOperator(
    task_id="retrain_task",
    python_callable=retrain_model,
    dag=dag,
)
