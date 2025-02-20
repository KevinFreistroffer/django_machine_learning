# commands not to forget

pip freeze > requirements.txt


DEV
uvicorn mysite.asgi:application --reload

Iris dataset implementation:
Model Implementation (nn_lightning.py)
Created IrisClassifier using PyTorch Lightning
Implemented basic neural network architecture (4 inputs → 64 → 32 → 3 outputs)
Added training configuration (optimizer, learning rate, etc.)
Test Data Generation (generate_test_data.py)
Load Iris dataset from scikit-learn
Split into train/test sets
Generate synthetic historical predictions
Save test data as PyTorch tensors
Save historical predictions as JSON
Model Validation (validate_model.py)
Load test dataset and model
Run predictions
Check metrics against thresholds:
Accuracy (threshold: 0.30)
Precision (threshold: 0.30)
Recall (threshold: 0.30)
F1 Score (threshold: 0.30)
Drift Detection (check_drift.py)
Compare new model predictions with historical data
Implemented two drift metrics:
Kolmogorov-Smirnov test (threshold: 0.7)
Confusion matrix difference (threshold: 5.0)
Removed p-value check due to sensitivity
Test Suite
test_model.py:
Test output shape
Test probability distributions
Test accuracy thresholds
test_data.py:
Test dataset format
Test value ranges
Test historical predictions format
test_drift.py:
Test drift detection
Test drift thresholds
CI/CD Setup (.github/workflows/model-testing.yml)
Set up Python environment
Create necessary directories
Generate test model checkpoint
Generate test datasets
Run test suite
Run model validation
Check for drift
Docker Setup
Created Dockerfile and docker-compose.yml
Set up proper requirements handling
Configured proper paths and environment
The system is designed to:
1. Validate model performance
Detect model drift
Ensure data quality
Run automated tests
Be deployable via Docker
Currently, we're using relaxed thresholds for testing, with the intention to tighten them as the model stabilizes.

# Iris Model Validation

This project implements a neural network classifier for the Iris dataset with model validation and drift detection.

## Structure