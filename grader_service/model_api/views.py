from django.http import JsonResponse
from .inference.model import load_model, grade_submission
import subprocess

model = load_model()

def grade_code(request):
    code = request.GET.get("code", "print('Hello')")
    criteria = request.GET.get("criteria", "Should greet user")
    result = grade_submission(code, criteria, model=model)
    return JsonResponse(result)


def trigger_retrain(request):
    try:
        result = subprocess.run(
            ["python", "/raid/ganesh/saurav/hari/grader_service/retrain.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        return JsonResponse(
            {
                "status": "success",
                "message": "Retraining started",
                "output": result.stdout,
            }
        )
    except subprocess.CalledProcessError as e:
        return JsonResponse(
            {"status": "error", "message": str(e), "error": e.stderr}, status=500
        )
