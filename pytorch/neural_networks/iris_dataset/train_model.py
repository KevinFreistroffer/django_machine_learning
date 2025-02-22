import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisNN
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data, augment_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, EPOCHS, LEARNING_RATE

def train_and_save_model():
    """Train the model and save it to the checkpoint path"""
    # Load data with validation set
    X_train, X_val, y_train, y_val = load_and_preprocess_data(return_val=True)
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(
        X_train.numpy(), 
        y_train.numpy(),
        noise_factor=0.01,  # Even less noise
        n_synthetic=5      # More synthetic samples
    )
    
    # Convert augmented data to tensors
    X_train_aug = torch.FloatTensor(X_train_aug)
    y_train_aug = torch.LongTensor(y_train_aug)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_aug, y_train_aug)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )
    
    # Initialize model
    model = IrisNN()
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_acc',  # Monitor validation accuracy
        mode='max',
        patience=20,
        min_delta=0.001,
        verbose=True
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS * 4,  # Even more epochs
        callbacks=[early_stopping],
        enable_progress_bar=True,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2,
        precision=32,
        deterministic=True,
        accelerator='auto',
        devices=1
    )
    
    # Train model with validation
    trainer.fit(model, train_loader, val_loader)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    trainer.save_checkpoint(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model() 