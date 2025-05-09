from django.http import JsonResponse
from .inference.model import load_model, grade_submission
import subprocess
import threading
import logging
import sqlite3
from datetime import datetime
import uuid

model = load_model()
logger = logging.getLogger(__name__)
current_model_version = None  # Track current model version


def init_model_version():
    """Initialize the current model version from MLflow."""
    global current_model_version
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            "CodeGraderModel", stages=["Production"]
        )
        if latest_versions:
            current_model_version = latest_versions[0].version
    except Exception as e:
        logger.error(f"Failed to initialize model version: {e}")


init_model_version()


def grade_code(request):
    code = request.GET.get("code", "print('Hello')")
    criteria = request.GET.get("criteria", "Should greet user")
    result = grade_submission(code, criteria, model=model)
    result["model_version"] = current_model_version  # Include model version in response
    return JsonResponse(result)


def trigger_retrain(request):
    def run_retrain(run_id):
        try:
            result = subprocess.run(
                ["python", "/raid/ganesh/saurav/hari/grader_service/retrain.py"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Retraining completed: {result.stdout}")
            update_retrain_status(run_id, "completed")
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


def retrain_status(request):
    run_id = request.GET.get("run_id")
    if not run_id:
        return JsonResponse(
            {"status": "error", "message": "run_id required"}, status=400
        )

    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status, error, start_time FROM retrain_status WHERE run_id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return JsonResponse(
            {"status": "error", "message": "Run ID not found"}, status=404
        )

    status, error, start_time = row
    response = {
        "run_id": run_id,
        "status": status,
        "start_time": start_time,
        "error": error,
        "deployed_model_version": current_model_version,
    }
    return JsonResponse(response)


def save_retrain_status(run_id, status, error=None):
    """Save retraining status to database."""
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS retrain_status
           (run_id TEXT PRIMARY KEY, status TEXT, error TEXT, start_time TEXT)"""
    )
    cursor.execute(
        "INSERT OR REPLACE INTO retrain_status (run_id, status, error, start_time) VALUES (?, ?, ?, ?)",
        (run_id, status, error, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def update_retrain_status(run_id, status, error=None):
    """Update retraining status in database."""
    conn = sqlite3.connect("grading_data.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE retrain_status SET status = ?, error = ? WHERE run_id = ?",
        (status, error, run_id),
    )
    conn.commit()
    conn.close()
