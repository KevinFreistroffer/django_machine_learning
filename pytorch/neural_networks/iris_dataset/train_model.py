import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisClassifier
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, EPOCHS, LEARNING_RATE

def train_and_save_model():
    """Train the model and save it to the checkpoint path"""
    # Load data
    X_train, y_train = load_and_preprocess_data(test_mode=False)
    
    # Initialize model with hyperparameters
    hparams = {
        'learning_rate': LEARNING_RATE
    }
    model = IrisClassifier(hparams=hparams)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Train model
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    model.train()
    for epoch in range(EPOCHS):
        outputs = model(X_train)
        loss = model.criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            best_state = {
                'state_dict': model.state_dict(),
                'hparams': hparams,
                'loss': best_loss
            }
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    
    # Save best model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(best_state, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH} with loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_and_save_model() 