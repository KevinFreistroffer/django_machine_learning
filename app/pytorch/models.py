from django.db import models
from django.utils import timezone

# Create your models here.

class HousePrediction(models.Model):
    file_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(default=timezone.now)
    prediction = models.FloatField(null=True, blank=True)
    input_features = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Prediction for {self.file_name} at {self.uploaded_at}"

    class Meta:
        ordering = ['-uploaded_at']
