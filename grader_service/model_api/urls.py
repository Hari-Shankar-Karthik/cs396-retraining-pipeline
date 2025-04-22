from django.urls import path
from .views import grade_code

urlpatterns = [
    path("grade/", grade_code),
]
