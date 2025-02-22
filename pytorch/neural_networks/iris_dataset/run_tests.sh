#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root directory
cd "${SCRIPT_DIR}/../../../"

# Train and save model
python -m pytorch.neural_networks.iris_dataset.train_model

# Run model validation
python -m pytorch.neural_networks.iris_dataset.validate_model

# Run unit tests
python -m unittest discover -s pytorch/neural_networks/iris_dataset/tests -p "test_*.py" 