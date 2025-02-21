from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch

def load_and_preprocess_data(test_mode=False, return_all=False):
    """
    This is like getting our wine collection ready for the robot to learn from!
    We:
    1. Get all our wine data (like a big spreadsheet of wine info)
    2. Clean it up (like organizing messy wine bottles)
    3. Split it into practice wines and test wines
    4. Make all the numbers easy for our robot to understand
    """
    # Load wine quality dataset (you can use UCI wine quality dataset)
    wine = load_wine()
    X = wine.data
    
    # Generate more stable quality scores (between 3 and 9)
    # Use features to generate scores for more realistic relationships
    base_scores = np.mean(X[:, :3], axis=1)  # Use first 3 features
    base_scores = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
    y = base_scores * 6 + 3  # Scale to range [3, 9]
    
    # Print sample information before scaling
    print("\nData Statistics Before Scaling:")
    print(f"X shape: {X.shape}")
    print(f"X range: {X.min():.2f} to {X.max():.2f}")
    print(f"y range: {y.min():.2f} to {y.max():.2f}")
    
    if return_all:
        # Scale all data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    # Split our wines into practice and test bottles
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Make all our measurements use the same scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Print statistics after scaling
    print("\nData Statistics After Scaling:")
    print(f"X_train range: {X_train.min():.2f} to {X_train.max():.2f}")
    print(f"X_test range: {X_test.min():.2f} to {X_test.max():.2f}")
    print(f"y_train range: {y_train.min():.2f} to {y_train.max():.2f}")
    print(f"y_test range: {y_test.min():.2f} to {y_test.max():.2f}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    if test_mode:
        return X_test, y_test
    
    return X_train, X_test, y_train, y_test, scaler

def print_sample_info(features, quality, feature_names):
    """
    This is like looking at one wine bottle's label to understand its properties
    """
    print("\nFirst wine in dataset:")
    print("-" * 50)
    wine_data = {
        name: f"{value:.4f}" for name, value in zip(feature_names, features)
    }
    wine_data['quality_score'] = f"{quality:.1f}"
    print("Wine data = {")
    for key, value in wine_data.items():
        print(f"    '{key}': {value},")
    print("}")
    print("-" * 50)

def augment_data(X, y, noise_factor=0.05, n_synthetic=3):
    """
    This is like creating more wine samples by slightly changing the ones we have!
    Like adding a tiny bit more or less of each ingredient to see what happens.
    """
    X_augmented = [X]
    y_augmented = [y]
    
    # Add small random changes to existing wines
    for _ in range(n_synthetic):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        y_noisy = y + np.random.normal(0, 0.1, y.shape)  # Small changes to scores
        X_augmented.append(X_noisy)
        y_augmented.append(y_noisy)
    
    # Mix similar wines together
    for i in range(len(X)):
        for j in range(i + 1, min(i + 3, len(X))):
            # Only mix wines with similar scores
            if abs(y[i] - y[j]) < 1.0:
                alpha = np.random.random()
                X_mixed = alpha * X[i] + (1 - alpha) * X[j]
                y_mixed = alpha * y[i] + (1 - alpha) * y[j]
                X_augmented.append(X_mixed.reshape(1, -1))
                y_augmented.append([y_mixed])

    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)
    
    # Shuffle all our wines
    shuffle_idx = np.random.permutation(len(X_augmented))
    return X_augmented[shuffle_idx], y_augmented[shuffle_idx] 