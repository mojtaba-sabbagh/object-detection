from django.urls import path
from .views import DetectView, HealthView

urlpatterns = [
    path('detect/', DetectView.as_view(), name='detect'),
    path('health/', HealthView.as_view(), name='health'),
]
