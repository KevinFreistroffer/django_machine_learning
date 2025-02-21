import torch
import numpy as np
from sklearn.inspection import permutation_importance
from pytorch.neural_networks.wine_quality.data_utils import load_and_preprocess_data
from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.config import MODEL_PATH
import matplotlib.pyplot as plt
import pandas as pd

def analyze_feature_importance():
    """Analyze which wine characteristics matter most for quality prediction"""
    # Load data and model
    X_test, y_test = load_and_preprocess_data(test_mode=True)
    model = WineQualityRegressor.load_from_checkpoint(MODEL_PATH)
    model.eval()
    
    # Convert model predictions to sklearn-compatible format
    def model_predict(X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            return model(X_tensor).cpu().numpy().flatten()
    
    # Calculate feature importance
    result = permutation_importance(
        model_predict, X_test.numpy(), y_test.numpy(),
        n_repeats=10, random_state=42
    )
    
    # Plot importance scores
    feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']  # Add all features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Wine Feature Importance for Quality Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

if __name__ == "__main__":
    analyze_feature_importance() 