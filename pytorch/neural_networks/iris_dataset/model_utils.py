import torch
import os
from .nn_lightning import IrisClassifier
from torch.utils.data import DataLoader, TensorDataset
from .config import BATCH_SIZE
from sklearn.preprocessing import StandardScaler
from .nn_lightning import IrisNN

def load_model(model_path):
    """Load the trained model from checkpoint"""
    try:
        # Create a new model instance
        model = IrisNN()
        
        # Load state dict with weights_only=True
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_predictions(model, X_test, y_test=None):
    """Get predictions from model"""
    model.eval()  # Ensure model is in eval mode
    
    # Convert to tensor if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
    
    if y_test is not None:
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.LongTensor(y_test)
    else:
        y_test = torch.zeros_like(predictions)
    
    return predictions, y_test 