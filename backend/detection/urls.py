# detection/urls.py
from django.urls import path
from .views import DetectView, HealthView, BatchDetectView

urlpatterns = [
    path('detect/', DetectView.as_view(), name='detect'),
    path('detect/batch/', BatchDetectView.as_view(), name='detect-batch'),  # NEW
    path('health/', HealthView.as_view(), name='health'),
]
