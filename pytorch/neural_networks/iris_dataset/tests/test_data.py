import pytest
import torch
import os
import json
import numpy as np
from ..data_utils import load_and_preprocess_data

@pytest.fixture
def test_dataset():
    """Load test dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_dataset.pt')
    return torch.load(data_path, weights_only=True)

@pytest.fixture
def historical_predictions():
    """Load historical predictions"""
    pred_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_predictions.json')
    with open(pred_path, 'r') as f:
        return json.load(f)

def test_dataset_shape(test_dataset):
    """Test if dataset has correct shape"""
    assert len(test_dataset['X_test'].shape) == 2, "Features should be 2D"
    assert len(test_dataset['y_test'].shape) == 1, "Labels should be 1D"
    assert len(test_dataset['X_test']) == len(test_dataset['y_test']), "X and y should have same length"

def test_dataset_values():
    """Test that dataset values are within expected ranges"""
    X_test, y_test = load_and_preprocess_data(test_mode=True)
    
    # Convert to numpy for easier testing
    X_np = X_test.numpy()
    y_np = y_test.numpy()
    
    # Test feature ranges (Iris features are standardized, so can be negative)
    assert np.all(np.abs(X_np) < 5), "Feature values should be within reasonable range"
    
    # Test labels
    assert np.all((y_np >= 0) & (y_np <= 2)), "Labels should be between 0 and 2"
    assert len(np.unique(y_np)) == 3, "Should have 3 unique classes"

def test_historical_predictions_format(historical_predictions):
    """Test if historical predictions file has correct format"""
    # Check main structure
    required_keys = ['predictions', 'confusion_matrix', 'metadata']
    assert all(key in historical_predictions for key in required_keys), "Missing required keys in historical predictions"
    
    # Check metadata format
    metadata_keys = ['timestamp', 'model_version', 'dataset_size']
    assert all(key in historical_predictions['metadata'] for key in metadata_keys), "Missing required metadata keys"
    
    # Check predictions format
    assert isinstance(historical_predictions['predictions'], list), "Predictions should be a list"
    if historical_predictions['predictions']:  # If not empty
        assert all(isinstance(pred, int) for pred in historical_predictions['predictions']), "All predictions should be integers"
        assert all(0 <= pred <= 2 for pred in historical_predictions['predictions']), "All predictions should be 0, 1, or 2"
    
    # Check confusion matrix format
    assert isinstance(historical_predictions['confusion_matrix'], list), "Confusion matrix should be a list"
    assert len(historical_predictions['confusion_matrix']) == 3, "Should be a 3x3 matrix"
    for row in historical_predictions['confusion_matrix']:
        assert len(row) == 3, "Each row should have 3 elements"
        assert all(isinstance(x, int) for x in row), "Matrix elements should be integers" 