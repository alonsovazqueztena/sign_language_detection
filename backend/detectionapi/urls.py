# detectionapi/urls.py
from django.urls import path
from .views import detect_sign_language

urlpatterns = [
    path('detect/', detect_sign_language, name='detect_sign_language'),
]
