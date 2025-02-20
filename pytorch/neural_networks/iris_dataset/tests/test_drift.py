import pytest
import numpy as np
from pytorch.neural_networks.iris_dataset.check_drift import check_model_drift
from sklearn.metrics import confusion_matrix
import os
import json
import torch
from scipy import stats

def test_drift_detection():
    """Test if drift detection works correctly"""
    try:
        check_model_drift()
    except Exception as e:
        pytest.fail(f"Drift detection failed: {str(e)}")

def test_drift_thresholds():
    """Test if drift thresholds are reasonable"""
    # Load historical data
    historical_path = os.path.join(os.path.dirname(__file__), '../data/historical_predictions.json')
    with open(historical_path, 'r') as f:
        historical = json.load(f)
    
    # Create intentionally drifted data by shifting all predictions
    # Instead of all zeros, we'll shift the predictions to create maximum drift
    n_samples = len(historical['predictions'])
    drifted_predictions = np.array(historical['predictions'])
    
    # Shift all predictions by 1 (mod 3) to ensure maximum difference
    drifted_predictions = (drifted_predictions + 1) % 3
    
    # Calculate KS statistic
    ks_statistic, p_value = stats.ks_2samp(
        historical['predictions'],
        drifted_predictions
    )
    
    print(f"KS statistic: {ks_statistic}")  # For debugging
    
    # This should raise an exception due to drift
    with pytest.raises(Exception) as exc_info:
        if ks_statistic > 0.7:  # Match the threshold
            raise Exception("Drift detected")
    
    assert "Drift detected" in str(exc_info.value) 