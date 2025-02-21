import pytest
import numpy as np
from sklearn.metrics import confusion_matrix
import json
from scipy import stats
from ..config import KS_THRESHOLD, MATRIX_DIFF_THRESHOLD

def test_drift_detection(test_data_path, generate_test_data):
    """Test if drift detection works correctly"""
    from ..check_drift import check_model_drift
    try:
        check_model_drift()
    except Exception as e:
        pytest.fail(f"Drift detection failed: {str(e)}")

def test_drift_thresholds(test_data_path, generate_test_data):
    """Test if drift thresholds are reasonable"""
    historical = generate_test_data
    
    # Create intentionally drifted data
    n_samples = len(historical['predictions'])
    drifted_predictions = []
    
    # Create maximum drift by alternating between classes
    for i in range(n_samples):
        drifted_predictions.append((i % 3))  # Cycle through 0,1,2
    
    # Create drifted confusion matrix
    drifted_conf_matrix = confusion_matrix(
        historical['predictions'],
        drifted_predictions
    ).tolist()
    
    # Calculate drift metrics
    ks_statistic, _ = stats.ks_2samp(
        historical['predictions'],
        drifted_predictions
    )
    
    matrix_diff = np.abs(
        np.array(historical['confusion_matrix']) - 
        np.array(drifted_conf_matrix)
    ).mean()
    
    print(f"KS statistic: {ks_statistic}")
    print(f"Matrix difference: {matrix_diff}")
    
    # This should raise an exception due to drift
    with pytest.raises(Exception) as exc_info:
        if ks_statistic > KS_THRESHOLD or matrix_diff > MATRIX_DIFF_THRESHOLD:
            raise Exception(
                f"Drift detected:\n"
                f"KS statistic: {ks_statistic:.3f} (threshold: {KS_THRESHOLD})\n"
                f"Matrix difference: {matrix_diff:.3f} (threshold: {MATRIX_DIFF_THRESHOLD})"
            )
    
    assert "Drift detected" in str(exc_info.value) 