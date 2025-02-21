import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from pytorch.neural_networks.wine_quality.nn_lightning import WineQualityRegressor
from pytorch.neural_networks.wine_quality.data_utils import load_and_preprocess_data

def train_model():
    """
    This is like running a wine tasting school for our robot!
    We:
    1. Get the wine data ready
    2. Set up the training program
    3. Let the robot practice tasting wines
    4. Save the robot's knowledge for later
    """
    # Get our wine data ready
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Package the data into batches (like organizing wine bottles into cases)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    val_dataset = TensorDataset(X_test, y_test)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )
    
    # Create our wine tasting robot
    model = WineQualityRegressor(input_size=X_train.shape[1])
    
    # Set up callbacks (like having expert wine tasters watching and giving advice)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )
    
    # Save the best version of our robot
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='wine_model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    # Set up logging to track progress
    logger = TensorBoardLogger("lightning_logs", name="wine_quality")
    
    # Create the trainer (like setting up the wine tasting school)
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=5,
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
        precision=32,
        deterministic=True
    )
    
    # Start training! (Let the robot taste and learn about wines)
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final version of our robot
    trainer.save_checkpoint("models/wine_model_final.ckpt")
    
    print(f"Best R² score achieved: {model.best_r2:.4f}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("lightning_logs", exist_ok=True)
    
    train_model() 