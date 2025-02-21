from django import forms
from .models import WineQuality

class WineQualityForm(forms.ModelForm):
    class Meta:
        model = WineQuality
        fields = [
            'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 
            'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
            'proanthocyanins', 'color_intensity', 'hue', 'od280_od315',
            'proline', 'actual_quality'
        ]
        widgets = {
            field: forms.NumberInput(attrs={'class': 'form-control'})
            for field in fields
        } 