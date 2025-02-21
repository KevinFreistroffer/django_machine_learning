"""
Think of this like a control panel for our wine-tasting robot!
All the important settings and rules are kept here so they're easy to find and change.
"""

import os

# Get base directory - handle both test and production environments
if 'PYTEST_CURRENT_TEST' in os.environ:
    # In test environment
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    # In production environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (like maps showing where to find everything)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'wine_model_final.ckpt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'wine_scaler.npy')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_dataset.pt')
HISTORICAL_PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'historical_predictions.json')

# Training settings (like instructions for teaching the robot)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
WEIGHT_DECAY = 0.01

# Model architecture (like the robot's brain design)
INPUT_SIZE = 11
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.2
LEAKY_RELU_SLOPE = 0.1

# Validation thresholds (like passing grades for our robot)
R2_THRESHOLD = 0.5     # At least 50% of variance explained
RMSE_THRESHOLD = 1.0   # Within 1 point on average
MAE_THRESHOLD = 0.8    # Less than 0.8 points average absolute error

# Drift detection thresholds (like warning signs for when the robot gets confused)
KS_THRESHOLD = 0.1           # Maximum allowed distribution difference
MEAN_DIFF_THRESHOLD = 0.5    # Maximum allowed change in average predictions
STD_DIFF_THRESHOLD = 0.3     # Maximum allowed change in prediction spread

# Data augmentation settings (like rules for creating practice examples)
NOISE_FACTOR = 0.05    # How much random variation to add
N_SYNTHETIC = 3        # How many variations to make of each wine
SIMILAR_WINE_THRESHOLD = 1.0  # Maximum score difference for mixing wines

# Random seed for reproducibility (like making sure the robot learns the same way each time)
RANDOM_SEED = 42 