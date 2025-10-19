from django.contrib import admin
from .models import YoloModel

@admin.register(YoloModel)
class YoloModelAdmin(admin.ModelAdmin):
    list_display = ("name", "base_model", "date_built", "num_params", "map", "map_5095", "size", "is_active")
    list_filter = ("is_active", "base_model", "date_built")
    search_fields = ("name", "base_model", "weights_path")
    actions = ["make_active"]

    @admin.action(description="Mark selected model as ACTIVE (only one will remain active)")
    def make_active(self, request, queryset):
        # Activate the most recent selection; deactivate others handled by model.save()
        obj = queryset.first()
        if obj:
            obj.is_active = True
            obj.save()
