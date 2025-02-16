from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import HouseDataForm
from .models import HousePrediction
import pandas as pd
import json
import torch
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and other functional operations


class HousePriceModel(nn.Module):
    def __init__(self, input_features):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

def index(request):
    """
    Main view for PyTorch application
    """
    context = {
        'title': 'PyTorch App',
        'welcome_message': 'Welcome to the PyTorch Application!'
    }
    return render(request, 'pytorch/index.html', context)

def house_prediction(request):
    """
    View for house price prediction
    """
    if request.method == 'POST':
        form = HouseDataForm(request.POST)
        if form.is_valid():
            try:
                print("Form is valid")
                # Get form data and convert to DataFrame - modified to use single values
                form_data = {
                    field: form.cleaned_data[field]  # Remove the list wrapper
                    for field in form.cleaned_data
                }
                # Convert to DataFrame with a single row
                df = pd.DataFrame([form_data])  # Wrap the dictionary in a list
                
                # Get features (all columns except 'Price' if present)
                features = df.values
                
                # Convert features to tensor
                features_tensor = torch.FloatTensor(features)
                
                model = HousePriceModel(input_features=features.shape[1])
                checkpoint = torch.load('media/house_price_model.ckpt')
                state_dict = checkpoint['state_dict']
                
                new_state_dict = {}
                for key in state_dict:
                    new_key = key.replace('model.', '')
                    new_state_dict[new_key] = state_dict[key]
                
                model.load_state_dict(new_state_dict)
                model.eval()
                
                # Make predictions
                with torch.no_grad():
                    predictions = model(features_tensor)
                    predictions = predictions.numpy()
                
                # Update the prediction assignment
                form_data['Predicted_Price'] = float(predictions[0][0])  # Convert to regular float
                print(form_data)
                return render(request, 'pytorch/prediction_result.html', {
                    'form': form,
                    'result': form_data,
                    'prediction_made': True
                })
                
            except Exception as e:
                print(f"Error processing form: {str(e)}")
                form.add_error(None, f"Error processing form: {str(e)}")
        else:
            print("Form is not valid")
    else:
        form = HouseDataForm()
    
    return render(request, 'pytorch/house_prediction.html', {'form': form})

def prediction_result(request, pk):
    """
    View for displaying prediction results
    """
    prediction = HousePrediction.objects.get(pk=pk)
    return render(request, 'pytorch/prediction_result.html', {
        'prediction': prediction,
        'title': 'Prediction Result'
    })
