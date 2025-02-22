from django.shortcuts import render, redirect
from django.contrib import messages
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys
from pathlib import Path
from sklearn.datasets import load_wine

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .forms import WineQualityForm
from .models import WineQuality
from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.config import MODEL_PATH

def wine_quality_view(request):
    # Get recent predictions
    predictions = WineQuality.objects.order_by('-created_at')[:10]
    
    # Get sample test data
    wine = load_wine()
    test_data = []
    
    # Get stats for display
    stats = {
        'total_samples': len(wine.data),
        'quality_ranges': {
            'Premium (7-9)': 0,
            'Average (5-6)': 0,
            'Below Average (3-4)': 0
        }
    }
    
    # Use more data points (similar to Iris dataset)
    for i in range(30):  # Increased from 5 to 30 samples
        quality = np.random.uniform(3, 9)
        
        # Update stats
        if quality >= 7:
            stats['quality_ranges']['Premium (7-9)'] += 1
        elif quality >= 5:
            stats['quality_ranges']['Average (5-6)'] += 1
        else:
            stats['quality_ranges']['Below Average (3-4)'] += 1
            
        row = {
            'alcohol': wine.data[i][0],
            'malic_acid': wine.data[i][1],
            'ash': wine.data[i][2],
            'alcalinity_of_ash': wine.data[i][3],
            'magnesium': wine.data[i][4],
            'total_phenols': wine.data[i][5],
            'flavanoids': wine.data[i][6],
            'nonflavanoid_phenols': wine.data[i][7],
            'proanthocyanins': wine.data[i][8],
            'color_intensity': wine.data[i][9],
            'hue': wine.data[i][10],
            'od280_od315': wine.data[i][11],
            'proline': wine.data[i][12],
            'actual_quality': quality,
            'quality_class': 'Premium' if quality >= 7 else 'Average' if quality >= 5 else 'Below Average'
        }
        test_data.append(row)
    
    # Sort by quality for better display
    test_data.sort(key=lambda x: x['actual_quality'], reverse=True)
    
    if request.method == 'POST':
        form = WineQualityForm(request.POST)
        if form.is_valid():
            # Save form but don't commit yet
            wine = form.save(commit=False)
            
            # Get features for prediction
            features = np.array([[
                wine.alcohol, wine.malic_acid, wine.ash,
                wine.alcalinity_of_ash, wine.magnesium,
                wine.total_phenols, wine.flavanoids,
                wine.nonflavanoid_phenols, wine.proanthocyanins,
                wine.color_intensity, wine.hue,
                wine.od280_od315, wine.proline
            ]])
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Load model and make prediction
            model = WineQualityRegressor.load_from_checkpoint(MODEL_PATH)
            model.eval()
            
            # Get prediction with uncertainty
            with torch.no_grad():
                X = torch.FloatTensor(features_scaled)
                mean_pred, std_pred = model.predict_with_uncertainty(X)
                
                wine.predicted_quality = mean_pred.item()
                wine.prediction_confidence = 1 - std_pred.item()
            
            wine.save()
            messages.success(request, 'Wine quality prediction made successfully!')
            
            # Add prediction details for template
            prediction = {
                'predicted_quality': wine.predicted_quality,
                'features': {
                    'alcohol': wine.alcohol,
                    'malic_acid': wine.malic_acid,
                    'ash': wine.ash,
                    'alcalinity_of_ash': wine.alcalinity_of_ash,
                    'magnesium': wine.magnesium,
                    'total_phenols': wine.total_phenols,
                    'flavanoids': wine.flavanoids,
                    'nonflavanoid_phenols': wine.nonflavanoid_phenols,
                    'proanthocyanins': wine.proanthocyanins,
                    'color_intensity': wine.color_intensity,
                    'hue': wine.hue,
                    'od280_od315': wine.od280_od315,
                    'proline': wine.proline
                },
                'error': abs(wine.predicted_quality - wine.actual_quality)
            }
            
            return render(request, 'pages/wine_quality/predict.html', {
                'form': form,
                'predictions': predictions,
                'test_data': test_data,
                'prediction': prediction,
                'stats': stats
            })
    else:
        form = WineQualityForm()
    
    return render(request, 'pages/wine_quality/predict.html', {
        'form': form,
        'predictions': predictions,
        'test_data': test_data,
        'stats': stats
    })
