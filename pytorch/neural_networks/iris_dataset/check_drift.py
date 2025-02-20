import os
import torch
from sklearn.metrics import confusion_matrix
import json
from .model_utils import load_model, get_predictions
from .metrics import calculate_drift_metrics
from .config import *
from torch.utils.data import TensorDataset, DataLoader

def check_model_drift():
    """Check for model drift using historical predictions"""
    # We're going to compare how our model used to perform versus how it performs now
    # to make sure it hasn't gotten worse over time
    
    # Load the old predictions we saved from before
    historical_path = os.path.join(os.path.dirname(__file__), HISTORICAL_PREDICTIONS_PATH)
    with open(historical_path, 'r') as f:
        historical = json.load(f)
    
    # Get some test data ready that we'll use to check our model
    data_path = os.path.join(os.path.dirname(__file__), TEST_DATA_PATH)
    test_data = torch.load(data_path)
    
    # Package our test data in a way that PyTorch likes
    test_dataset = TensorDataset(test_data['X_test'], test_data['y_test'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Load our current model and use it to make predictions on our test data
    model_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)
    model = load_model(model_path)
    new_predictions, labels = get_predictions(model, test_loader)
    
    # Create a "report card" showing how many predictions were right and wrong
    new_conf_matrix = confusion_matrix(
        test_data['y_test'],
        new_predictions
    ).tolist()
    
    # Compare the old and new performance using special math
    drift_metrics = calculate_drift_metrics(
        historical['predictions'],
        new_predictions,
        historical['confusion_matrix'],
        new_conf_matrix
    )
    
    # If the model has gotten too much worse, raise an alarm!
    if (drift_metrics['ks_statistic'] > KS_THRESHOLD or 
        drift_metrics['matrix_diff'] > MATRIX_DIFF_THRESHOLD):
        raise Exception(f"""
        Model drift detected:
        KS statistic: {drift_metrics['ks_statistic']:.3f} (threshold: {KS_THRESHOLD})
        Confusion matrix difference: {drift_metrics['matrix_diff']:.3f} (threshold: {MATRIX_DIFF_THRESHOLD})
        """)
    
    # If we get here, everything is okay!
    print("No significant model drift detected")

if __name__ == "__main__":
    check_model_drift() 