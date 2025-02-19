import pytest
import torch
import os
import json

@pytest.fixture
def test_dataset():
    data_path = os.path.join(os.path.dirname(__file__), '../data/test_dataset.pt')
    return torch.load(data_path)

@pytest.fixture
def historical_predictions():
    hist_path = os.path.join(os.path.dirname(__file__), '../data/historical_predictions.json')
    with open(hist_path, 'r') as f:
        return json.load(f)

def test_dataset_shape(test_dataset):
    """Test if dataset has correct shape and features"""
    assert test_dataset['X_test'].shape[1] == 4, "Iris dataset should have 4 features"
    assert len(test_dataset['X_test']) == len(test_dataset['y_test']), "X and y should have same length"

def test_dataset_values(test_dataset):
    """Test if dataset values are in expected range"""
    X = test_dataset['X_test']
    y = test_dataset['y_test']
    
    assert torch.all(X >= 0), "Feature values should be non-negative"
    assert torch.all(y >= 0) and torch.all(y < 3), "Labels should be 0, 1, or 2"

def test_historical_predictions_format(historical_predictions):
    """Test if historical predictions file has correct format"""
    required_keys = ['predictions', 'confusion_matrix', 'metadata']
    assert all(key in historical_predictions for key in required_keys), "Missing required keys in historical predictions"
    
    metadata_keys = ['timestamp', 'model_version', 'dataset_size']
    assert all(key in historical_predictions['metadata'] for key in metadata_keys), "Missing required metadata keys" 