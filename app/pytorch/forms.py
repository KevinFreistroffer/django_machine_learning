from django import forms

class HouseDataCSVForm(forms.Form):
    csv_file = forms.FileField(
        label='Select a CSV file',
        help_text='Upload a CSV file with house features',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    ) 

class HouseDataForm(forms.Form):
    MedInc = forms.FloatField(label='Median Income')
    HouseAge = forms.FloatField(label='House Age')
    AveRooms = forms.FloatField(label='Average Number of Rooms')
    AveBedrms = forms.FloatField(label='Average Number of Bedrooms')
    Population = forms.IntegerField(label='Population')
    AveOccup = forms.IntegerField(label='Average Occupancy')
    Latitude = forms.FloatField(label='Latitude')
    Longitude = forms.FloatField(label='Longitude')

