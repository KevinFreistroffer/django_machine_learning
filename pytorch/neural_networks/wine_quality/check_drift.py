import torch
import numpy as np
from scipy import stats
import json
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
    MODEL_PATH, HISTORICAL_PREDICTIONS_PATH,
    KS_THRESHOLD, MEAN_DIFF_THRESHOLD, STD_DIFF_THRESHOLD
)

def check_model_drift():
    """
    Let's make sure our wine-tasting robot hasn't gotten confused over time!
    We'll compare its current predictions with its past ones to see if it's still consistent.
    """
    # Load historical predictions
    hist_path = os.path.join(os.path.dirname(__file__), HISTORICAL_PREDICTIONS_PATH)
    with open(hist_path, 'r') as f:
        historical = json.load(f)
    
    # Get current predictions
    _, X_test, _, y_test, _ = load_and_preprocess_data()
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model = WineQualityRegressor.load_from_checkpoint(model_path)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        current_preds = model(X_test)
        current_preds = current_preds.cpu().float().numpy().flatten()
    
    # Calculate drift metrics
    ks_stat, p_value = stats.ks_2samp(historical['predictions'], current_preds)
    mean_diff = np.abs(np.mean(historical['predictions']) - np.mean(current_preds))
    std_diff = np.abs(np.std(historical['predictions']) - np.std(current_preds))
    
    # Use thresholds from config
    if (ks_stat > KS_THRESHOLD or 
        mean_diff > MEAN_DIFF_THRESHOLD or 
        std_diff > STD_DIFF_THRESHOLD):
        raise Exception(
            f"Model drift detected!\n"
            f"KS Statistic: {ks_stat:.3f} (threshold: {KS_THRESHOLD})\n"
            f"Mean Difference: {mean_diff:.3f} (threshold: {MEAN_DIFF_THRESHOLD})\n"
            f"Std Difference: {std_diff:.3f} (threshold: {STD_DIFF_THRESHOLD})"
        )
    
    print("\nNo significant drift detected! 🎉")

if __name__ == "__main__":
    check_model_drift() 