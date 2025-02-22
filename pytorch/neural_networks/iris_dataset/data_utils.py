import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(test_mode=False):
    """Load and preprocess the Iris dataset"""
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # First standardize all data to ensure consistent scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use stratified sampling and shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y,
        shuffle=True
    )
    
    # Additional step to ensure similar distributions
    # Randomly shuffle within each class to reduce drift
    for class_label in np.unique(y):
        train_mask = y_train == class_label
        test_mask = y_test == class_label
        
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(X_train[train_mask])
        np.random.shuffle(X_test[test_mask])
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Return appropriate data based on mode
    if test_mode:
        return X_test_tensor, y_test_tensor
    else:
        return X_train_tensor, y_train_tensor

def print_sample_info(features, label, feature_names, target_names):
    """Shows us what one flower looks like in our data, like looking at
    one card in a deck of flower cards"""
    print("\nFirst row of dataset:")
    print("-" * 50)
    row_data = {
        name: f"{value:.4f}" for name, value in zip(feature_names, features)
    }
    row_data['target_class'] = str(target_names[label])
    print("Row data = {")
    for key, value in row_data.items():
        print(f"    '{key}': {value},")
    print("}")
    print("-" * 50)

def augment_data(X, y, noise_factor=0.05, n_synthetic=3):
    """Makes more flower data by playing pretend!
    
    We do this in two ways:
    1. We take our real flowers and make copies with tiny changes
       (like drawing the same flower multiple times, each slightly different)
    2. We mix two similar flowers together to make a new one
       (like mixing red and blue paint to make purple)"""
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
                alpha = np.random.random()
                interpolated = alpha * class_samples[i] + (1 - alpha) * class_samples[j]
                X_augmented.append(interpolated.reshape(1, -1))
                y_augmented.append([class_idx])

    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_augmented))
    return X_augmented[shuffle_idx], y_augmented[shuffle_idx] 