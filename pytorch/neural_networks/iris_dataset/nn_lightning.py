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
            self.learning_rate = 0.01
        else:
            self.learning_rate = hparams.get('learning_rate', 0.01)
        
        # Define model layers with a better architecture
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
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class IrisNN(pl.LightningModule):
    def __init__(self):
        super(IrisNN, self).__init__()
        # Our network is like a flower expert that learns to identify iris types
        # It looks at 4 measurements (length/width of petals and sepals)
        self.fc1 = nn.Linear(4, 64)  
        # Helps keep all measurements on a similar scale, like using the same ruler
        self.bn1 = nn.BatchNorm1d(64)
        
        # Takes the first analysis and refines it further
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        # Makes the final decision: which of the 3 iris types it sees
        self.fc3 = nn.Linear(32, 3)
        
        # Helps the network learn subtle patterns
        self.relu = nn.LeakyReLU(0.1)
        # Randomly ignores some information during training to prevent memorization
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        # Gives the network a smart starting point instead of random guesses
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # This is like an assembly line for processing the flower measurements
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))  # First processing step
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))  # Second processing step
        x = self.fc3(x)  # Final decision making
        return x

    def training_step(self, batch, batch_idx):
        # This is where the network learns from each batch of flowers
        # It makes predictions and adjusts itself when it makes mistakes
        X, y = batch
        logits = self(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics for monitoring overfitting
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Tests how well the network does on flowers it hasn't seen during training
        # Helps us know if it's actually learning or just memorizing
        X, y = batch
        logits = self(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics for monitoring overfitting
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self._f1_score(preds, y), on_epoch=True, prog_bar=True)
        
        # Calculate and log the gap between train and val metrics
        self.log('acc_gap', self.trainer.callback_metrics.get('train_acc', 0) - acc, 
                on_epoch=True, prog_bar=True)
        
        return loss

    def _f1_score(self, preds, targets, eps=1e-8):
        # Gives us a balanced score of how well we're doing
        # Makes sure we're good at identifying all types of irises, not just some
        # For each type of flower, we count:
        tp = torch.zeros(3).to(preds.device)  # When we said "it's this flower" and we were right!
        fp = torch.zeros(3).to(preds.device)  # Oops! We said "it's this flower" but we were wrong
        fn = torch.zeros(3).to(preds.device)  # We missed this flower when we should have found it
        
        for i in range(3):
            tp[i] = ((preds == i) & (targets == i)).sum()
            fp[i] = ((preds == i) & (targets != i)).sum()
            fn[i] = ((preds != i) & (targets == i)).sum()
        
        precision = tp / (tp + fp + eps)  # How many of our positive predictions were correct
        recall = tp / (tp + fn + eps)     # How many actual positives did we catch
        f1 = 2 * (precision * recall) / (precision + recall + eps)  # Combines both measures
        return f1.mean()

    def configure_optimizers(self):
        # Sets up how the network learns:
        # - How big steps it takes when learning
        # - How it adjusts its learning speed over time
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=0.01,
            amsgrad=True
        )
        
        # Calculate steps_per_epoch based on dataset size and batch size
        steps_per_epoch = len(X_train) // 32  # batch_size = 32
        
        # Learning schedule
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=0.001,
                epochs=200,
                steps_per_epoch=steps_per_epoch,  # Use calculated value
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            ),
            "interval": "step",
            "frequency": 1
        }
        
        return [self.optimizer], [scheduler]

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