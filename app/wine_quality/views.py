from django.shortcuts import render, redirect
from django.contrib import messages
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from .forms import WineQualityForm
from .models import WineQuality
from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.config import MODEL_PATH

def wine_quality_view(request):
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
            return redirect('wine_quality')
    else:
        form = WineQualityForm()
    
    # Get recent predictions
    predictions = WineQuality.objects.order_by('-created_at')[:10]
    
    return render(request, 'wine_quality/wine_quality.html', {
        'form': form,
        'predictions': predictions
    })
