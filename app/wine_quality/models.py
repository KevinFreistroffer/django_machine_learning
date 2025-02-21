from django.db import models

# Create your models here.

class WineQuality(models.Model):
    """
    Store wine quality predictions and actual values
    """
    # Features
    alcohol = models.FloatField()
    malic_acid = models.FloatField()
    ash = models.FloatField()
    alcalinity_of_ash = models.FloatField()
    magnesium = models.FloatField()
    total_phenols = models.FloatField()
    flavanoids = models.FloatField()
    nonflavanoid_phenols = models.FloatField()
    proanthocyanins = models.FloatField()
    color_intensity = models.FloatField()
    hue = models.FloatField()
    od280_od315 = models.FloatField(verbose_name='OD280/OD315')
    proline = models.FloatField()
    
    # Prediction and actual
    predicted_quality = models.FloatField(null=True)
    actual_quality = models.FloatField()
    prediction_confidence = models.FloatField(null=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Wine Quality Predictions"
    
    def __str__(self):
        return f"Wine {self.id} - Predicted: {self.predicted_quality:.1f}, Actual: {self.actual_quality:.1f}"
