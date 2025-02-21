"""Configuration settings for Iris model"""

import os

# Get base directory - handle both test and production environments
if 'PYTEST_CURRENT_TEST' in os.environ:
    # In test environment
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    # In production environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model validation thresholds
ACCURACY_THRESHOLD = 0.30
PRECISION_THRESHOLD = 0.30
RECALL_THRESHOLD = 0.30
F1_THRESHOLD = 0.30

# Drift detection thresholds
KS_THRESHOLD = 0.05
MATRIX_DIFF_THRESHOLD = 3.0

# Data paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'iris_model.ckpt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'iris_scaler.npy')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_dataset.pt')
HISTORICAL_PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'historical_predictions.json')

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 200 