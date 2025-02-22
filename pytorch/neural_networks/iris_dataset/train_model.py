import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from pytorch.neural_networks.iris_dataset.nn_lightning import IrisNN
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data, augment_data
from pytorch.neural_networks.iris_dataset.config import MODEL_PATH, EPOCHS, LEARNING_RATE

def train_and_save_model():
    """Train the model and save it to the checkpoint path"""
    # Load data
    X_train, y_train = load_and_preprocess_data(test_mode=False)
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(
        X_train.numpy(), 
        y_train.numpy(),
        noise_factor=0.05,
        n_synthetic=2
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
    
    # Initialize model
    model = IrisNN()  # Using the better architecture
    
    # Create optimizer with better learning rate schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Train model with more epochs
    model.train()
    for epoch in range(EPOCHS * 2):  # Double the epochs
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = torch.nn.CrossEntropyLoss()(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        accuracy = 100. * correct / total
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS*2}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS * 2
        }
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model() 