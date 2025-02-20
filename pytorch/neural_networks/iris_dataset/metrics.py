from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from scipy import stats

def calculate_model_metrics(y_true, y_pred):
    """
    Let's check how well our model is doing! 
    We compare what the model thought (y_pred) with what was actually true (y_true).
    It's like giving the model a report card with different scores:
    - Accuracy: How many times the model got the answer right
    - Precision: When the model says "this is a cat", how often is it really a cat?
    - Recall: Out of all the real cats, how many did the model find?
    - F1: A special score that combines precision and recall into one number
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_drift_metrics(historical_preds, new_preds, hist_conf_matrix, new_conf_matrix):
    """
    Sometimes our model gets confused over time, like how a child might forget 
    things they learned last year. This function helps us spot when that happens!
    
    We do this by:
    1. Playing "spot the difference" between old predictions and new ones 
       (that's what the KS test does)
    2. Looking at how much the model's behavior has changed by comparing its 
       old and new "answer patterns" (that's what the matrix difference shows)
    """
    # KS test
    ks_statistic, _ = stats.ks_2samp(historical_preds, new_preds)
    
    # Confusion matrix difference
    matrix_diff = np.abs(
        np.array(hist_conf_matrix) - np.array(new_conf_matrix)
    ).mean()
    
    return {
        'ks_statistic': ks_statistic,
        'matrix_diff': matrix_diff
    } 