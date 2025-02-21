import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.data_utils import load_and_preprocess_data
from pytorch.neural_networks.wine_quality.config import (
    MODEL_PATH, R2_THRESHOLD, RMSE_THRESHOLD, MAE_THRESHOLD
)

def validate_model():
    """
    Let's check how good our wine-tasting robot really is!
    We'll:
    1. Give it some wines it hasn't seen before
    2. Compare its scores with real expert scores
    3. Calculate how accurate it is in different ways
    """
    # Load test data
    _, X_test, _, y_test, _ = load_and_preprocess_data()
    
    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model = WineQualityRegressor.load_from_checkpoint(model_path)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = y_pred.cpu().float().numpy().flatten()
    y_test = y_test.cpu().float().numpy().flatten()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 50)
    print(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"True values range: {y_test.min():.2f} to {y_test.max():.2f}")
    print(f"Mean prediction: {y_pred.mean():.2f}")
    print(f"Mean true value: {y_test.mean():.2f}")
    print(f"Prediction std: {y_pred.std():.2f}")
    print(f"True value std: {y_test.std():.2f}")
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Check if model meets requirements
    print("\nModel Validation Results:")
    print("-" * 50)
    print(f"R² Score: {r2:.3f} (threshold: {R2_THRESHOLD})")
    print(f"RMSE: {rmse:.3f} (threshold: {RMSE_THRESHOLD})")
    print(f"MAE: {mae:.3f} (threshold: {MAE_THRESHOLD})")
    
    # Print some example predictions
    print("\nSample Predictions:")
    print("-" * 50)
    for true, pred in zip(y_test[:5], y_pred[:5]):
        print(f"True: {true:.1f}, Predicted: {pred:.1f}, Error: {abs(true-pred):.1f}")
    
    # Validate against thresholds
    assert r2 >= R2_THRESHOLD, f"R² score {r2:.3f} below threshold {R2_THRESHOLD}"
    assert rmse <= RMSE_THRESHOLD, f"RMSE {rmse:.3f} above threshold {RMSE_THRESHOLD}"
    assert mae <= MAE_THRESHOLD, f"MAE {mae:.3f} above threshold {MAE_THRESHOLD}"
    
    print("\nModel validation passed! 🎉")

if __name__ == "__main__":
    validate_model() 