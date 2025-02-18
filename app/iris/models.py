from django.db import models
from django.utils import timezone

# Create your models here.

class IrisPrediction(models.Model):
    IRIS_CLASSES = [
        ('setosa', 'Iris Setosa'),
        ('versicolor', 'Iris Versicolor'),
        ('virginica', 'Iris Virginica'),
    ]
    
    sepal_length = models.FloatField()
    sepal_width = models.FloatField()
    petal_length = models.FloatField()
    petal_width = models.FloatField()
    predicted_class = models.CharField(max_length=20, choices=IRIS_CLASSES)
    actual_class = models.CharField(
        max_length=20, 
        choices=IRIS_CLASSES,
        default='versicolor'
    )
    prediction_date = models.DateTimeField(default=timezone.now)
    
    @property
    def is_correct(self):
        return self.predicted_class == self.actual_class
    
    def __str__(self):
        return f"Iris Prediction: {self.predicted_class} (Actual: {self.actual_class}) at {self.prediction_date}"
