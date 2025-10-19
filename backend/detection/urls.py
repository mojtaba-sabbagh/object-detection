from django.urls import path
from . import views

urlpatterns = [
    path("detect/", views.DetectView.as_view(), name="detect"),
    path("detect/batch/", views.BatchDetectView.as_view(), name="detect-batch"),
    path("model/current/", views.CurrentModelView.as_view(), name="model-current"),
    path("health/", views.HealthView.as_view(), name="health"),
]
