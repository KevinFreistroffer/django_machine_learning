import torch
import os
from .nn_lightning import IrisClassifier

def load_model(model_path):
    """
    Think of this like opening a saved coloring book! We're getting back our
    special flower-sorting machine that we saved earlier.
    """
    try:
        # First, we try to open it like a fancy coloring book (Lightning style)
        model = IrisClassifier.load_from_checkpoint(model_path)
    except Exception:
        # Oops! If that doesn't work, we'll open it like a regular coloring book
        model = IrisClassifier()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    return model

def get_predictions(model, data_loader):
    """
    This is like giving our flower-sorting machine a bunch of flowers and
    writing down what type it thinks each flower is!
    """
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    return all_preds, all_labels 