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
    
    # Augment training data with careful class balancing
    X_train_aug, y_train_aug = augment_data(
        X_train.numpy(), 
        y_train.numpy(),
        noise_factor=0.01,
        n_synthetic=5  # Moderate augmentation
    )
    
    # Convert augmented data back to tensors
    X_train_aug = torch.FloatTensor(X_train_aug)
    y_train_aug = torch.LongTensor(y_train_aug)
    
    # Create data loaders with smaller batch size for better generalization
    train_loader = DataLoader(
        TensorDataset(X_train_aug, y_train_aug),
        batch_size=8,  # Smaller batch size
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=len(X_val)  # Use full batch for validation
    )
    
    # Initialize model with specific hyperparameters
    model = IrisClassifier({
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout': 0.2
    })
    
    # Configure trainer with early stopping and model checkpointing
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=20,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.dirname(MODEL_PATH),
        filename='iris_best',
        save_top_k=1,
        mode='min',
    )
    
    # Train with gradient clipping and learning rate monitoring
    trainer = pl.Trainer(
        max_epochs=200,  # Train longer
        accelerator='auto',
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=0.5,  # Add gradient clipping
        log_every_n_steps=1,
        deterministic=True,  # For reproducibility
        accumulate_grad_batches=4  # Effective batch size of 32
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Load best model and save it
    best_model = IrisClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(best_model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train_and_save_model() 