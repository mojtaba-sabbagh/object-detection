from django.db import models
from django.utils import timezone

class YoloModel(models.Model):
    name = models.CharField(max_length=100, unique=True)  # model name (display)
    date_built = models.DateField(default=timezone.now)
    base_model = models.CharField(max_length=50)          # e.g. "yolo11n"
    num_params = models.BigIntegerField(help_text="Number of parameters")
    map = models.FloatField(null=True, blank=True)
    map_5095 = models.FloatField(null=True, blank=True)
    size = models.CharField(max_length=50, help_text="e.g. '23.1 MB'")
    weights_path = models.CharField(max_length=512, help_text="Absolute path to .pt/.onnx")
    is_active = models.BooleanField(default=False, db_index=True)

    class Meta:
        ordering = ["-is_active", "-date_built", "name"]

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Enforce single active model
        if self.is_active:
            YoloModel.objects.exclude(pk=self.pk).filter(is_active=True).update(is_active=False)

    def __str__(self):
        return f"{self.name} ({'ACTIVE' if self.is_active else 'inactive'})"
