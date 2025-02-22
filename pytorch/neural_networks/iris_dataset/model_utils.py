import torch
import os
from .nn_lightning import IrisNN
from torch.utils.data import DataLoader, TensorDataset
from .config import BATCH_SIZE

def load_model(model_path):
    """
    Think of this like opening a saved coloring book! We're getting back our
    special flower-sorting machine that we saved earlier.
    """
    try:
        # First, we try to open it like a fancy coloring book (Lightning style)
        model = IrisNN.load_from_checkpoint(model_path)
    except Exception:
        # Oops! If that doesn't work, we'll open it like a regular coloring book
        model = IrisNN()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    return model

def get_predictions(model, X_test, y_test=None):
    """Get predictions from model"""
    model.eval()
    
    # Convert input to DataLoader if it's not already
    if not isinstance(X_test, DataLoader):
        if y_test is not None:
            dataset = TensorDataset(X_test, y_test)
        else:
            dataset = TensorDataset(X_test, torch.zeros(len(X_test)))  # Dummy labels
        X_test = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in X_test:
            x, y = batch
            outputs = model(x)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu())
            all_labels.extend(y.cpu())
    
    return torch.tensor(all_predictions), torch.tensor(all_labels) 