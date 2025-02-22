import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing
import numpy as np
import torch.nn.functional as F
from pytorch.neural_networks.iris_dataset.config import LEARNING_RATE, MODEL_PATH, SCALER_PATH
from pytorch.neural_networks.iris_dataset.data_utils import load_and_preprocess_data

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Print sample information
print("\nFirst row of dataset:")
print("-" * 50)
first_row = {
    name: f"{value:.4f}" for name, value in zip(data.feature_names, X[0])
}
first_row['target_class'] = str(data.target_names[y[0]])
print("Row data = {")
for key, value in first_row.items():
    print(f"    '{key}': {value},")
print("}")
print("-" * 50)

# After loading and before scaling the data
def augment_data(X, y, noise_factor=0.05, n_synthetic=3):
    # This function makes our dataset bigger and more diverse by:
    # 1. Adding small random changes to existing flower measurements
    # 2. Creating new samples by mixing measurements from similar flowers
    X_augmented = [X]
    y_augmented = [y]
    
    # Add noise to existing samples
    for _ in range(n_synthetic):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y)
    
    # Interpolation between same-class samples
    for class_idx in range(3):
        class_samples = X[y == class_idx]
        for i in range(len(class_samples)):
            for j in range(i + 1, min(i + 3, len(class_samples))):
                # Create interpolated sample
                alpha = np.random.random()
                interpolated = alpha * class_samples[i] + (1 - alpha) * class_samples[j]
                X_augmented.append(interpolated.reshape(1, -1))
                y_augmented.append([class_idx])

    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)
    
    # Shuffle the augmented dataset
    shuffle_idx = np.random.permutation(len(X_augmented))
    return X_augmented[shuffle_idx], y_augmented[shuffle_idx]

# Apply augmentation before scaling
X, y = augment_data(X, y)
print(f"\nDataset size after augmentation: {len(X)} samples")

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class IrisClassifier(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters(hparams)
        if hparams is None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = hparams.get('learning_rate', 0.001)
        
        # Match the saved checkpoint architecture
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 3)
        )
        
        # Define loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            amsgrad=True
        )
        return optimizer

class IrisNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Match the saved model structure
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=0.01,
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

def main():
    # The main training program:
    # 1. Organizes the flower data into batches
    # 2. Creates and trains the network
    # 3. Saves the trained network
    # 4. Tests it on some example flowers
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )

    # Create model
    model = IrisNN()

    # Early stopping callback with more careful monitoring
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min',
        min_delta=1e-4  # Minimum change to qualify as an improvement
    )

    # Set up the training process
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        precision="16-mixed",
        deterministic=True,
        # Add validation checks every epoch
        check_val_every_n_epoch=1,
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Save the model and scaler
    trainer.save_checkpoint(MODEL_PATH)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    
    # Save scaler data as a dictionary
    scaler_data = {
        'mean_': scaler.mean_,
        'scale_': scaler.scale_
    }
    np.save(SCALER_PATH, scaler_data)
    
    # Print sample predictions
    model.eval()
    with torch.no_grad():
        sample_input = X_test[:5]
        predictions = model(sample_input)
        pred_classes = torch.argmax(predictions, dim=1)
        
        print("\nSample Predictions:")
        print("-" * 50)
        for i in range(5):
            print(f"True: {data.target_names[y_test[i]]}, Predicted: {data.target_names[pred_classes[i]]}")

if __name__ == '__main__':
    # On Windows, this line prevents the program from getting stuck in an infinite loop
    # when using multiple processes. It's like telling all the worker processes "hey, you're
    # helpers, not the boss!" so they don't try to run the main program again and again.
    multiprocessing.freeze_support()
    main()