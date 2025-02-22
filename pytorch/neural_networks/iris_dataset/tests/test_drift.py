import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[4])
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
import numpy as np
from scipy import stats
from pytorch.neural_networks.iris_dataset.config import KS_THRESHOLD, MATRIX_DIFF_THRESHOLD
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data

class TestDataDrift(unittest.TestCase):
    def setUp(self):
        self.X_train, _ = load_and_preprocess_data(test_mode=False)
        self.X_test, _ = load_and_preprocess_data(test_mode=True)
        
    def test_drift(self):
        """Test for data drift between train and test sets"""
        # Convert to numpy for statistical tests
        train_data = self.X_train.numpy()
        test_data = self.X_test.numpy()
        
        # Compare distributions of each feature
        for i in range(train_data.shape[1]):
            statistic, _ = stats.ks_2samp(train_data[:, i], test_data[:, i])
            self.assertLess(statistic, KS_THRESHOLD, 
                f"Feature {i} shows significant drift: KS statistic = {statistic:.3f}")
        
        # Compare correlation matrices
        train_corr = np.corrcoef(train_data.T)
        test_corr = np.corrcoef(test_data.T)
        matrix_diff = np.abs(train_corr - test_corr).mean()
        self.assertLess(matrix_diff, MATRIX_DIFF_THRESHOLD,
            f"Correlation structure shows significant drift: diff = {matrix_diff:.3f}")

if __name__ == '__main__':
    unittest.main() 