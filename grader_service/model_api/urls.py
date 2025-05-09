from django.urls import path
from .views import grade_code, trigger_retrain, retrain_status

urlpatterns = [
    path("grade/", grade_code),
    path("retrain/", trigger_retrain),
    path("retrain_status/", retrain_status),
]
