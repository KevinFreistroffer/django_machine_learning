#!/bin/bash

# Get the directory of this script and the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Set Python path to include project root
export PYTHONPATH="${PROJECT_ROOT}"

# Change to project root directory
cd "${PROJECT_ROOT}"

# Train model first
echo "Training model..."
PYTHONPATH="${PROJECT_ROOT}" python -m pytorch.neural_networks.iris_dataset.train_model

# Run tests
echo "Running tests..."
PYTHONPATH="${PROJECT_ROOT}" python -m pytest "${SCRIPT_DIR}/tests/" --cov="${SCRIPT_DIR}" --cov-report=xml 