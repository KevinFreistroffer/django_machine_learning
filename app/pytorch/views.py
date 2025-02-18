from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import HouseDataForm
from .models import HousePrediction
import pandas as pd
import json
import torch
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Activation functions and other functional operations
import numpy as np
from sklearn.preprocessing import StandardScaler


class HousePriceModel(nn.Module):
    def __init__(self, input_features):
        super(HousePriceModel, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden layers
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        # Output layer
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.dropout1(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(F.relu(self.fc2(x))))
        x = self.dropout3(self.bn3(F.relu(self.fc3(x))))
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
        print("Form Data:", request.POST)
        if form.is_valid():
            try:
                print("Form is valid")
                # Get form data and convert to DataFrame
                form_data = {
                    field: [float(form.cleaned_data[field])]  # Convert all values to float
                    for field in form.cleaned_data
                }
                df = pd.DataFrame(form_data)
                
                # Get features (all columns except 'Price' if present)
                features = df.values.astype(np.float32)  # Explicitly convert to float32
                
                # Convert features to tensor
                features_tensor = torch.FloatTensor(features)
                
                # Load model with weights_only=False since we need the full state dict
                model = HousePriceModel(input_features=features.shape[1])
                checkpoint = torch.load('media/house_price_model.ckpt', weights_only=False)
                # Load scalers with allow_pickle=True
                scaler_data = np.load("media/scalers.npy", allow_pickle=True).item()
                
                # Reconstruct the scaler
                scaler_y = StandardScaler()
                scaler_y.mean_ = scaler_data['scaler_y_mean']
                scaler_y.scale_ = scaler_data['scaler_y_scale']
                
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
                    predictions = scaler_y.inverse_transform(predictions.numpy())
                
                # Scale the prediction to $100,000s
                scaled_prediction = float(predictions[0][0]) * 100000
                form_data['Predicted_Price'] = scaled_prediction
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
            print("Form validation errors:", form.errors)
            print("Form non-field errors:", form.non_field_errors())
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
