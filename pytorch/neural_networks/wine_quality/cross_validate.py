import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.data_utils import load_and_preprocess_data, augment_data
from pytorch.neural_networks.wine_quality.config import (
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE
)

def cross_validate_model(n_folds=5):
    """Perform cross-validation to get robust performance estimates"""
    # Load all data
    X, y = load_and_preprocess_data(return_all=True)
    X, y = X.numpy(), y.numpy()
    
    # Setup cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    print(f"\nStarting {n_folds}-fold cross-validation")
    print("-" * 50)
    
    # Train and evaluate on each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Augment training data
        X_train_aug, y_train_aug = augment_data(
            X_train, y_train,
            noise_factor=0.05,
            n_synthetic=2
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_aug)
        y_train_tensor = torch.FloatTensor(y_train_aug)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            persistent_workers=True
        )
        
        # Initialize model
        model = WineQualityRegressor(input_size=X.shape[1])
        
        # Setup trainer
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            mode='min'
        )
        
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[early_stopping],
            enable_progress_bar=True,
            log_every_n_steps=5,
            gradient_clip_val=0.5,
            accelerator='auto',
            devices=1,
            deterministic=True
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_tensor)
            y_pred = y_pred.cpu().numpy().flatten()
        
        # Calculate metrics
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = np.mean(np.abs(y_val - y_pred))
        
        # Store metrics
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Fold {fold + 1} Results:")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
    
    # Print summary statistics
    print("\nCross-validation Summary:")
    print("-" * 50)
    print(f"R² Score: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
    print(f"MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    
    return {
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores
    }

if __name__ == "__main__":
    cross_validate_model() 