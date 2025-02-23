import pytest
import os
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

@pytest.fixture(scope="session")
def test_data_path():
    """Create and return paths for test data"""
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),  # tests directory
        '..',  # iris_dataset directory
    ))
    
    # Create necessary directories if they don't exist
    data_dir = os.path.join(base_path, 'data')
    models_dir = os.path.join(base_path, 'models')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    paths = {
        'base': base_path,
        'data': data_dir,
        'models': models_dir,
        'test_dataset': os.path.join(data_dir, 'test_dataset.pt'),
        'historical_predictions': os.path.join(data_dir, 'historical_predictions.json'),
        'model_checkpoint': os.path.join(models_dir, 'iris_model.ckpt'),
        'scaler': os.path.join(models_dir, 'iris_scaler.npy')
    }
    
    # Print paths for debugging
    print("\nTest paths:")
    for key, path in paths.items():
        print(f"{key}: {path}")
        print(f"Exists: {os.path.exists(path)}")
    
    return paths

@pytest.fixture(scope="session")
def generate_test_data(test_data_path):
    """Generate test data for unit tests"""
    from ..data_utils import load_and_preprocess_data
    
    # Generate test dataset in test mode
    X_test, y_test = load_and_preprocess_data(test_mode=True)
    
    # Save test dataset
    torch.save({
        'X_test': X_test,
        'y_test': y_test
    }, test_data_path['test_dataset'])
    
    # Generate sample predictions
    n_samples = len(y_test)
    predictions = []
    for i in range(n_samples):
        # 90% chance to predict correct class
        if np.random.random() < 0.9:
            predictions.append(int(y_test[i]))
        else:
            predictions.append(np.random.randint(0, 3))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions).tolist()
    
    # Save historical predictions
    historical_data = {
        'predictions': predictions,
        'confusion_matrix': conf_matrix,
        'metadata': {
            'timestamp': '2024-01-01',
            'model_version': '1.0.0',
            'dataset_size': n_samples
        }
    }
    
    with open(test_data_path['historical_predictions'], 'w') as f:
        json.dump(historical_data, f, indent=2)
    
    return historical_data

@pytest.fixture(scope="session")
def mock_model(test_data_path):
    """Create a mock model for testing"""
    from ..nn_lightning import IrisClassifier
    model = IrisClassifier()
    
    # Save mock model
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': 0,
        'global_step': 0
    }, test_data_path['model_checkpoint'])
    
    return model 