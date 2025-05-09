from django.http import JsonResponse
from .inference.model import load_model, grade_submission
import subprocess
import threading
import logging
import sqlite3
from datetime import datetime
import uuid
from deploy_model import deploy_model
import os

model = load_model()
logger = logging.getLogger(__name__)
current_model_version = None  # Track current model version


def init_model_version():
    global current_model_version
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()

        # Fetch the latest model versions from production
        latest_versions = client.get_latest_versions(
            "CodeGraderModel", stages=["Production"]
        )

        logger.info(f"Latest Production versions: {latest_versions}")

        if latest_versions:
            current_model_version = latest_versions[0].version
            logger.info(f"Current model version: {current_model_version}")
        else:
            logger.warning("No model in Production stage.")
    except Exception as e:
        logger.error(f"Failed to initialize model version: {e}")


init_model_version()


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


def grade_code(request):
    code = request.GET.get("code", "print('Hello')")
    criteria = request.GET.get("criteria", "Should greet user")
    result = grade_submission(code, criteria, model=model)
    result["model_version"] = current_model_version  # Include model version in response
    return JsonResponse(result)


def save_retrain_status(run_id, status, error=None, model_version=None):
    """Save retraining status to the database."""
    ensure_retrain_status_schema()
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO retrain_status (run_id, status, error, start_time, model_version) VALUES (?, ?, ?, ?, ?)",
        (run_id, status, error, datetime.now().isoformat(), model_version),
    )
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


def trigger_retrain(request):
    def run_retrain(run_id):
        try:
            result = subprocess.run(
                [
                    "python",
                    "/raid/ganesh/saurav/hari/grader_service/retrain.py",
                    run_id,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Retraining completed: {result.stdout}")
            update_retrain_status(run_id, "completed")  # Ensure completion status
        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed: {e.stderr}")
            update_retrain_status(run_id, "failed", str(e))

    try:
        run_id = str(uuid.uuid4())  # Unique ID for retraining run
        save_retrain_status(run_id, "running")
        thread = threading.Thread(target=run_retrain, args=(run_id,))
        thread.start()
        return JsonResponse(
            {"status": "success", "message": "Retraining started", "run_id": run_id}
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def deploy(request):
    run_id = request.GET.get("run_id")
    if not run_id:
        return JsonResponse(
            {"status": "error", "message": "run_id required"}, status=400
        )

    # Initialize the model version
    init_model_version()  # Ensures the current_model_version is updated

    ensure_retrain_status_schema()
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status, error, start_time, model_version FROM retrain_status WHERE run_id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return JsonResponse(
            {"status": "error", "message": "Run ID not found"}, status=404
        )

    status, error, start_time, model_version = row
    if status != "completed":
        return JsonResponse(
            {
                "run_id": run_id,
                "status": status,
                "start_time": start_time,
                "error": error,
                "deployed_model_version": current_model_version,
            }
        )

    # Retraining completed, deploy the model
    try:
        deploy_model()
        # Verify deployment
        if current_model_version == model_version:
            return JsonResponse(
                {
                    "status": "success",
                    "message": "Model deployed and ready",
                    "run_id": run_id,
                    "model_version": model_version,
                }
            )
        else:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Deployment failed: Model version mismatch",
                    "run_id": run_id,
                    "expected_version": model_version,
                    "deployed_version": current_model_version,
                },
                status=500,
            )
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return JsonResponse(
            {
                "status": "error",
                "message": f"Deployment failed: {str(e)}",
                "run_id": run_id,
            },
            status=500,
        )
