"""Configuration settings for Iris model"""

# Model validation thresholds
ACCURACY_THRESHOLD = 0.30
PRECISION_THRESHOLD = 0.30
RECALL_THRESHOLD = 0.30
F1_THRESHOLD = 0.30

# Drift detection thresholds
KS_THRESHOLD = 0.7
MATRIX_DIFF_THRESHOLD = 5.0

# Data paths
MODEL_PATH = 'models/iris_model.ckpt'
SCALER_PATH = 'models/iris_scaler.npy'
TEST_DATA_PATH = 'data/test_dataset.pt'
HISTORICAL_PREDICTIONS_PATH = 'data/historical_predictions.json'

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 200 