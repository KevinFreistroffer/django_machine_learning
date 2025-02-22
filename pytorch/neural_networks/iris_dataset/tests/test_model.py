import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
import torch
from pytorch.neural_networks.iris_dataset.model_utils import load_model, get_predictions
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, ACCURACY_THRESHOLD

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        self.X_test, self.y_test = load_and_preprocess_data(test_mode=True)
        self.model = load_model(MODEL_PATH)
    
    def test_model_accuracy(self):
        predictions, _ = get_predictions(self.model, self.X_test)
        accuracy = (predictions == self.y_test).float().mean().item()
        self.assertGreaterEqual(accuracy, ACCURACY_THRESHOLD, 
            f"Model accuracy {accuracy:.3f} below threshold {ACCURACY_THRESHOLD}")

if __name__ == '__main__':
    unittest.main() 