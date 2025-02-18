from django import forms

class IrisDataForm(forms.Form):
    sepal_length = forms.FloatField(
        label='Sepal Length (cm)',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    sepal_width = forms.FloatField(
        label='Sepal Width (cm)',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    petal_length = forms.FloatField(
        label='Petal Length (cm)',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    petal_width = forms.FloatField(
        label='Petal Width (cm)',
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    ) 