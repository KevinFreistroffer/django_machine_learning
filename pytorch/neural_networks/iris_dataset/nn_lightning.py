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

class IrisNN(pl.LightningModule):
    def __init__(self):
        super(IrisNN, self).__init__()
        # Network architecture
        self.fc1 = nn.Linear(4, 128)  # 4 input features
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.fc3 = nn.Linear(96, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 3)  # 3 output classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Slightly increased dropout

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout(self.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=10,
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

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.5,
        accumulate_grad_batches=2
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

    # Save the model and scaler
    trainer.save_checkpoint("models/iris_model.ckpt")
    np.save("models/iris_scaler.npy", {
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    })

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
    multiprocessing.freeze_support()
    main()