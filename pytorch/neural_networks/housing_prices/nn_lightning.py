import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing
import numpy as np

# Get information about house prices in California
data = fetch_california_housing()
X, y = data.data, data.target 
print("\nFirst row of dataset:")
print("-" * 50)
first_row = {
    name: f"{value:.4f}" for name, value in zip(data.feature_names, X[0])
}
first_row['target_price'] = f"{y[0]:.4f}"
print("Row data = {")
for key, value in first_row.items():
    print(f"    '{key}': {value},")
print("}")
print("-" * 50)

# Split our data into two parts:
# X = information about houses (like size, location, etc.)
# y = the actual house prices we want to predict

print("X", X)
print("y", y)
print(f"Number of samples in dataset: X shape = {X.shape}, y length = {len(y)}")

# Scale both features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split our data into two groups:
# - One group to teach our computer (training data, 80%)
# - One group to test how well it learned (testing data, 20%)
# We use 42 as a "seed" number so we get the same split every time we run the program
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert our data into a special format that PyTorch can understand
# It's like translating our numbers into a language the computer prefers
X_train = torch.tensor(X_train, dtype=torch.float32)
# Reshape our data so it's in the right format
# Like organizing items neatly in a container
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Now we're using PyTorch Lightning - it's like having a helpful assistant
# that organizes all our code and adds useful features automatically
class HousePriceNN(pl.LightningModule):
    def __init__(self):
        super(HousePriceNN, self).__init__()
        # Wider network with batch normalization
        self.fc1 = nn.Linear(X.shape[1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Increased dropout

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = nn.MSELoss()(y_hat, y)
        # Add R2 score monitoring
        r2 = 1 - loss / torch.var(y)
        self.log('train_loss', loss)
        self.log('train_r2', r2)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = nn.MSELoss()(y_hat, y)
        r2 = 1 - loss / torch.var(y)
        self.log('val_loss', loss)
        self.log('val_r2', r2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,  # More aggressive reduction
            patience=10,  # Reduced patience
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

def main():
    # Our data is still organized in the same way, but now Lightning helps manage it
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    # We also create a separate loader for testing
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        persistent_workers=True
    )

    # Create our model with improved training setup
    model = HousePriceNN()

    # Early stopping is now handled by Lightning
    # It's like having an assistant who knows when to stop training
    early_stopping = EarlyStopping(
        monitor='val_loss',    # Watch how well we're learning
        patience=50,           # Give up after 50 tries with no improvement
        verbose=True,          # Tell us what's happening
        mode='min'            # We want the loss to go down
    )

    # The Trainer is like a teaching manager
    # It organizes everything about how we train our model
    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=0.5,  # Add gradient clipping
        accumulate_grad_batches=4  # Accumulate gradients for larger effective batch size
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Save the model and scalers
    trainer.save_checkpoint("models/house_price_model.ckpt")
    
    # Save scalers using numpy arrays with a safer approach
    scaler_data = {
        'scaler_X_mean': scaler_X.mean_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_
    }
    np.save("models/scalers.npy", scaler_data)

    # Update the loading comment to match the new format
    """
    # To load scalers:
    scaler_data = np.load("models/scalers.npy", allow_pickle=True).item()
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.mean_ = scaler_data['scaler_X_mean']
    scaler_X.scale_ = scaler_data['scaler_X_scale']
    scaler_y.mean_ = scaler_data['scaler_y_mean']
    scaler_y.scale_ = scaler_data['scaler_y_scale']
    """

    print("\nFirst row of dataset:")
    print("-" * 50)
    for name, value in zip(data.feature_names, X[0]):
        print(f"{name:15} : {value:.4f}")
    print(f"{'Target price':15} : {y[0]:.4f}")
    print("-" * 50)

if __name__ == '__main__':
    # This is required for Windows
    multiprocessing.freeze_support()
    main()