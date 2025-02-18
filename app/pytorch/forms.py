from django import forms

class HouseDataCSVForm(forms.Form):
    csv_file = forms.FileField(
        label='Select a CSV file',
        help_text='Upload a CSV file with house features',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    ) 

class HouseDataForm(forms.Form):
    MedInc = forms.DecimalField(label='Median Income', max_digits=10, decimal_places=5, required=True)
    HouseAge = forms.DecimalField(label='House Age', max_digits=10, decimal_places=5, required=True)
    AveRooms = forms.DecimalField(label='Average Number of Rooms', max_digits=10, decimal_places=5, required=True)
    AveBedrms = forms.DecimalField(label='Average Number of Bedrooms', max_digits=10, decimal_places=5, required=True)
    Population = forms.IntegerField(label='Population', required=True)
    AveOccup = forms.DecimalField(label='Average Occupancy', max_digits=10, decimal_places=5, required=True)
    Latitude = forms.DecimalField(label='Latitude', max_digits=10, decimal_places=5, required=True)
    Longitude = forms.DecimalField(label='Longitude', max_digits=10, decimal_places=5, required=True)

