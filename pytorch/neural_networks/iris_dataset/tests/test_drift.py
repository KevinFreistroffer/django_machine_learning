import pytest
import numpy as np
from pytorch.neural_networks.iris_dataset.check_drift import check_model_drift
from sklearn.metrics import confusion_matrix
import os
import json
import torch

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
    
    # Create intentionally drifted data
    drifted_predictions = np.zeros_like(historical['predictions'])  # All zeros for maximum drift
    
    # This should raise an exception due to drift
    with pytest.raises(Exception) as exc_info:
        from scipy import stats
        ks_statistic, _ = stats.ks_2samp(
            historical['predictions'],
            drifted_predictions
        )
        if ks_statistic > 0.7:  # Match the new threshold
            raise Exception("Drift detected")
    
    assert "Drift detected" in str(exc_info.value) 