from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views

urlpatterns = [
    path("detect/", csrf_exempt(views.DetectView.as_view()), name="detect"),
    path("detect/batch/", csrf_exempt(views.BatchDetectView.as_view()), name="detect-batch"),
    path("model/current/", csrf_exempt(views.CurrentModelView.as_view()), name="model-current"),
    path("health/", views.HealthView.as_view(), name="health"),   # Health check usually doesn't need CSRF
]