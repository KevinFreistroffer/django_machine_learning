import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from pytorch.neural_networks.iris_dataset.nn_lightning import IrisClassifier
from iris_dataset.nn_lightning import IrisClassifier
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from .model_utils import load_model, get_predictions
from .metrics import calculate_model_metrics
from .config import *

def validate_model():
    # Let's get our test data from a special folder
    # Think of this like getting flashcards to quiz our robot
    data_path = os.path.join(os.path.dirname(__file__), TEST_DATA_PATH)
    test_data = torch.load(data_path)
    
    # Package our flashcards in a nice way so our robot can read them easily
    test_dataset = TensorDataset(test_data['X_test'], test_data['y_test'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Find and wake up our trained robot (that's our model!)
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model = load_model(model_path)
    
    # Let's see how well our robot can guess the answers
    predictions, labels = get_predictions(model, test_loader)
    metrics = calculate_model_metrics(labels, predictions)
    
    # Check if our robot is smart enough:
    # - Accuracy: How often it gets the answer right
    # - Precision: When it says "this is X", how often it's really X
    # - Recall: How good it is at finding ALL the X's
    # - F1: A special score that combines precision and recall
    assert metrics['accuracy'] >= ACCURACY_THRESHOLD, f"Accuracy {metrics['accuracy']:.2f} below threshold"
    assert metrics['precision'] >= PRECISION_THRESHOLD, f"Precision {metrics['precision']:.2f} below threshold"
    assert metrics['recall'] >= RECALL_THRESHOLD, f"Recall {metrics['recall']:.2f} below threshold"
    assert metrics['f1'] >= F1_THRESHOLD, f"F1 {metrics['f1']:.2f} below threshold"
    
    # Yay! Our robot passed all its tests! Let's see the scores:
    print("Model validation passed!")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")

if __name__ == "__main__":
    validate_model()