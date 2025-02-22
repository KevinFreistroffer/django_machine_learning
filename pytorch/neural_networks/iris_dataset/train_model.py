import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisClassifier
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data, augment_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, BATCH_SIZE

def train_and_save_model():
    """Train the model and save it"""
    # Set seeds for reproducibility
    pl.seed_everything(42, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load and preprocess data with validation split
    X_train, X_val, y_train, y_val = load_and_preprocess_data(return_val=True)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=len(X_val)
    )
    
    # Initialize model
    model = IrisClassifier()
    
    # Train with fixed settings
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        deterministic=True,
        enable_progress_bar=True
    )
    
    # Train and save
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train_and_save_model() 