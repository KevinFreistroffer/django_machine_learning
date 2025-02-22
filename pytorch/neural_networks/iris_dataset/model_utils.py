import torch
import os
from .nn_lightning import IrisClassifier
from torch.utils.data import DataLoader, TensorDataset
from .config import BATCH_SIZE
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load the trained model from checkpoint"""
    try:
        # Load checkpoint first
        checkpoint = torch.load(model_path, weights_only=True)
        
        # Create a new model instance
        model = IrisClassifier()
        
        # Clean and load state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove any unexpected keys
            state_dict = {k: v for k, v in state_dict.items() 
                         if k.startswith('model.') and not k.startswith('criterion')}
        else:
            state_dict = checkpoint
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set to evaluation mode
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
    
    # Standardize the test data
    scaler = StandardScaler()
    X_test_np = X_test.numpy()
    X_test_scaled = scaler.fit_transform(X_test_np)
    X_test = torch.FloatTensor(X_test_scaled)
    
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