from django.shortcuts import render
from .forms import IrisDataForm
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        # Network architecture
        self.fc1 = nn.Linear(4, 128)  # 4 input features
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.fc3 = nn.Linear(96, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 3)  # 3 output classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout(self.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x

def get_actual_class(features):
    """
    Determine the actual Iris class based on the classic rules:
    - Setosa: Petal length < 2.5 cm
    - Virginica: Petal length >= 4.9 cm and petal width >= 1.4 cm
    - Versicolor: Everything else
    """
    petal_length = features[0][2]
    petal_width = features[0][3]
    
    if petal_length < 2.5:
        return 'setosa'
    elif petal_length >= 4.9 and petal_width >= 1.4:
        return 'virginica'
    else:
        return 'versicolor'

def iris_prediction(request):
    prediction_result = None
    if request.method == 'POST':
        form = IrisDataForm(request.POST)
        if form.is_valid():
            try:
                # Get the input features
                features = np.array([[
                    form.cleaned_data['sepal_length'],
                    form.cleaned_data['sepal_width'],
                    form.cleaned_data['petal_length'],
                    form.cleaned_data['petal_width']
                ]])
                
                # Load the model
                model = IrisNN()
                checkpoint = torch.load('media/iris/iris_model.ckpt')
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                
                # Load and setup scaler
                scaler_data = np.load('media/iris/iris_scaler.npy', allow_pickle=True).item()
                scaler = StandardScaler()
                scaler.mean_ = scaler_data['scaler_mean']
                scaler.scale_ = scaler_data['scaler_scale']
                
                # Transform features
                features_scaled = scaler.transform(features)
                features_tensor = torch.FloatTensor(features_scaled)
                
                # Make prediction
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                classes = ['setosa', 'versicolor', 'virginica']
                probs = probabilities[0].numpy() * 100  # Convert to percentages
                
                # Determine actual class
                actual_class = get_actual_class(features)
                
                prediction_result = {
                    'class': classes[predicted.item()],
                    'actual_class': actual_class,
                    'features': form.cleaned_data,
                    'probabilities': {
                        classes[i]: f"{prob:.1f}%" 
                        for i, prob in enumerate(probs)
                    }
                }
                
                # Save prediction to database
                from .models import IrisPrediction
                IrisPrediction.objects.create(
                    sepal_length=features[0][0],
                    sepal_width=features[0][1],
                    petal_length=features[0][2],
                    petal_width=features[0][3],
                    predicted_class=classes[predicted.item()],
                    actual_class=actual_class
                )
                
            except Exception as e:
                form.add_error(None, f"Prediction error: {str(e)}")
    else:
        form = IrisDataForm()
    
    return render(request, 'iris_dataset/predict.html', {
        'form': form,
        'prediction': prediction_result,
        'title': 'Iris Species Prediction'
    })
