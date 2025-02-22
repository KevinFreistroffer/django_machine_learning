import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# Use absolute imports instead of relative imports
from pytorch.neural_networks.iris_dataset.model_utils import load_model, get_predictions
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, ACCURACY_THRESHOLD

def validate_model():
    """Validate the trained model meets performance requirements"""
    # Load test data
    X_test, y_test = load_and_preprocess_data(test_mode=True)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Get predictions
    predictions, _ = get_predictions(model, X_test)
    
    # Calculate accuracy
    accuracy = (predictions == y_test).float().mean().item()
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Validate accuracy meets threshold
    assert accuracy >= ACCURACY_THRESHOLD, f"Model accuracy {accuracy:.3f} below threshold {ACCURACY_THRESHOLD}"
    print("Model validation successful!")

if __name__ == "__main__":
    validate_model()