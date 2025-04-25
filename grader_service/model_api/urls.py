from django.urls import path
from .views import grade_code, trigger_retrain

urlpatterns = [
    path("grade/", grade_code),
    path("retrain/", trigger_retrain),
]
