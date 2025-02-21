from django.contrib import admin
from .models import WineQuality

@admin.register(WineQuality)
class WineQualityAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_quality', 'actual_quality', 'prediction_confidence', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('id', 'predicted_quality', 'actual_quality')
