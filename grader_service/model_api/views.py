from django.http import JsonResponse
from .inference.model import load_model, grade_submission

model = load_model()  # Cache the model once


def grade_code(request):
    code = request.GET.get("code", "print('Hello')")
    criteria = request.GET.get("criteria", "Should greet user")
    result = grade_submission(code, criteria, model=model)
    return JsonResponse(result)
