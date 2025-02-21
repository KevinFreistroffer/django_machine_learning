import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score
import numpy as np

class WineQualityRegressor(pl.LightningModule):
    """
    Imagine you're a wine taster! But instead of using your tongue,
    you're using a computer to guess how good a wine is based on its ingredients.
    This is like having a super-smart wine robot that learns to rate wines from 0 to 10.
    """
    def __init__(self, input_size=11):
        super().__init__()
        # Simpler architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Keep track of the best score
        self.best_r2 = -float('inf')
        
        # Save hyperparameters
        self.save_hyperparameters()

    def _init_weights(self):
        """Initialize weights for better starting point"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Simple forward pass"""
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y.view(-1, 1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.mse_loss(y_pred, y.view(-1, 1))
        
        # Calculate R² score
        y_pred_np = y_pred.cpu().detach().numpy().flatten()
        y_np = y.cpu().numpy().flatten()
        r2 = r2_score(y_np, y_pred_np)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_r2', r2, prog_bar=True)
        
        if r2 > self.best_r2:
            self.best_r2 = r2
            self.log('best_r2', self.best_r2, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def predict_with_uncertainty(self, x, n_samples=10):
        """Make predictions with uncertainty estimates"""
        self.train()  # Enable dropout for uncertainty estimation
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self(x)
                predictions.append(pred)
        
        # Calculate mean and std of predictions
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()  # Back to evaluation mode
        return mean_pred, std_pred
