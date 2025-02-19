import pytest
import torch
import os
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisClassifier
from sklearn.metrics import accuracy_score
import numpy as np

@pytest.fixture
def test_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/test_dataset.pt')
    return torch.load(data_path)

@pytest.fixture
def model():
    model_path = os.path.join(os.path.dirname(__file__), '../checkpoints/model.ckpt')
    return IrisClassifier.load_from_checkpoint(model_path)

def test_model_output_shape(model, test_data):
    """Test if model outputs correct shape"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data['X_test'])
    
    assert outputs.shape[1] == 3, "Model should output 3 classes"
    assert outputs.shape[0] == len(test_data['y_test']), "Batch size mismatch"

def test_model_output_values(model, test_data):
    """Test if model outputs valid probabilities"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data['X_test'])
        probs = torch.softmax(outputs, dim=1)
    
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities should be between 0 and 1"
    assert torch.allclose(torch.sum(probs, dim=1), torch.ones(probs.shape[0])), "Probabilities should sum to 1"

def test_model_accuracy(model, test_data):
    """Test if model meets minimum accuracy threshold"""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data['X_test'])
        predictions = torch.argmax(outputs, dim=1)
    
    accuracy = accuracy_score(test_data['y_test'], predictions)
    assert accuracy >= 0.85, f"Model accuracy {accuracy:.2f} below minimum threshold of 0.85" 