import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
import json
import os
import torch
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisClassifier

def check_model_drift():
    # Load historical predictions
    historical_path = os.path.join(os.path.dirname(__file__), 
                                 'data/historical_predictions.json')
    with open(historical_path, 'r') as f:
        historical = json.load(f)
    
    # Generate new predictions using current model and test data
    data_path = os.path.join(os.path.dirname(__file__), 'data/test_dataset.pt')
    test_data = torch.load(data_path)
    
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints/model.ckpt')
    
    try:
        # Try loading as Lightning checkpoint
        model = IrisClassifier.load_from_checkpoint(model_path)
    except Exception:
        # Fallback to loading as regular PyTorch checkpoint
        model = IrisClassifier()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(test_data['X_test'])
        new_predictions = torch.argmax(outputs, dim=1).numpy()
    
    new_conf_matrix = confusion_matrix(
        test_data['y_test'], 
        new_predictions
    ).tolist()
    
    new_data = {
        'predictions': new_predictions.tolist(),
        'confusion_matrix': new_conf_matrix
    }
    
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(
        historical['predictions'], 
        new_data['predictions']
    )
    
    # Check confusion matrix stability
    hist_conf = np.array(historical['confusion_matrix'])
    new_conf = np.array(new_data['confusion_matrix'])
    matrix_diff = np.abs(hist_conf - new_conf).mean()
    
    # Define drift thresholds
    KS_THRESHOLD = 0.7
    P_VALUE_THRESHOLD = 0.001
    MATRIX_DIFF_THRESHOLD = 5.0
    
    # Check for significant drift
    if (ks_statistic > KS_THRESHOLD or 
        p_value < P_VALUE_THRESHOLD or 
        matrix_diff > MATRIX_DIFF_THRESHOLD):
        raise Exception(f"""
        Model drift detected:
        KS statistic: {ks_statistic:.3f} (threshold: {KS_THRESHOLD})
        P-value: {p_value:.3f} (threshold: {P_VALUE_THRESHOLD})
        Confusion matrix difference: {matrix_diff:.3f} (threshold: {MATRIX_DIFF_THRESHOLD})
        """)
    
    print("No significant model drift detected")

if __name__ == "__main__":
    check_model_drift() 