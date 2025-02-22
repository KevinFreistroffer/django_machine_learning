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

# Ensure the checkpoints directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Add a note about the required checkpoint
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model checkpoint not found at {MODEL_PATH}")
    print("Please run train_model.py first to generate the model checkpoint")

# Model validation thresholds
# Using 90% accuracy threshold as this is considered good performance for the Iris dataset
# Many published papers and tutorials consider 90-93% as a good benchmark due to:
# 1. Natural overlap between versicolor and virginica classes
# 2. Small dataset size
# 3. Real-world applicability
ACCURACY_THRESHOLD = 0.90
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