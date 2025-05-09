from django.http import JsonResponse
from .inference.model import load_model, grade_submission
import subprocess
import threading
import logging

model = load_model()
logger = logging.getLogger(__name__)


def grade_code(request):
    code = request.GET.get("code", "print('Hello')")
    criteria = request.GET.get("criteria", "Should greet user")
    result = grade_submission(code, criteria, model=model)
    return JsonResponse(result)


def trigger_retrain(request):

    def run_retrain():
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

    try:
        # Run retraining in a background thread
        thread = threading.Thread(target=run_retrain)
        thread.start()
        return JsonResponse(
            {"status": "success", "message": "Retraining started in background"}
        )
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
