from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
from pytorch.neural_networks.wine_quality.data_utils import load_wine

def explain_prediction(model, input_x, target_class=None):
    """Explain why the model made a specific prediction"""
    # Get feature names
    feature_names = load_wine().feature_names
    
    # Calculate attributions
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_x, target=target_class)
    
    # Plot attribution scores
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, attributions.mean(dim=0).cpu().numpy())
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Contributions to Wine Quality Prediction')
    plt.tight_layout()
    plt.savefig('prediction_explanation.png') 