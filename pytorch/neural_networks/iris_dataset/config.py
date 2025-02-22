"""Configuration settings for Iris model"""

import os
from pathlib import Path

# Get the absolute path to the iris_dataset directory
IRIS_DIR = Path(__file__).resolve().parent

# Model paths with absolute paths
MODEL_PATH = str(IRIS_DIR / 'checkpoints' / 'model.ckpt')
TEST_DATA_PATH = str(IRIS_DIR / 'data' / 'test_dataset.pt')

# Create directories if they don't exist
os.makedirs(IRIS_DIR / 'checkpoints', exist_ok=True)
os.makedirs(IRIS_DIR / 'data', exist_ok=True)

# Set high accuracy threshold for educational purposes
# Note: For real-world Iris classification, 90%+ is typically considered good
ACCURACY_THRESHOLD = 0.95
PRECISION_THRESHOLD = 0.30
RECALL_THRESHOLD = 0.30
F1_THRESHOLD = 0.30

# Drift detection thresholds
KS_THRESHOLD = 0.20
MATRIX_DIFF_THRESHOLD = 0.2

# Data paths
SCALER_PATH = os.path.join(IRIS_DIR, 'models', 'iris_scaler.npy')
HISTORICAL_PREDICTIONS_PATH = os.path.join(IRIS_DIR, 'data', 'historical_predictions.json')

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 100 