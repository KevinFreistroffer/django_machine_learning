import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess_data(test_mode=False, return_val=False):
    """Load and preprocess the Iris dataset"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # First standardize all data to ensure consistent scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if return_val:
        # Split into train, validation, and test with stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        
        return X_train, X_val, y_train, y_val
    
    # Original split for test mode
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    if test_mode:
        return X_test, y_test
    return X_train, y_train

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

def augment_data(X, y, noise_factor=0.01, n_synthetic=3):
    """Makes more flower data by creating synthetic samples"""
    X_augmented = [X]
    y_augmented = [y]
    
    # Get class counts and indices
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_indices = {c: np.where(y == c)[0] for c in unique_classes}
    
    # For each class
    for class_idx in unique_classes:
        class_X = X[class_indices[class_idx]]
        
        # 1. Add noise to existing samples
        for _ in range(n_synthetic):
            # Calculate per-feature standard deviation
            feature_stds = np.std(class_X, axis=0)
            # Add proportional noise
            noise = np.random.normal(0, noise_factor * feature_stds, class_X.shape)
            X_noisy = class_X + noise
            X_augmented.append(X_noisy)
            y_augmented.append(np.full(len(class_X), class_idx))
        
        # 2. Create synthetic samples using k-nearest neighbors
        n_neighbors = min(5, len(class_X)-1)
        for idx in range(len(class_X)):
            # Find k nearest neighbors
            distances = np.linalg.norm(class_X - class_X[idx], axis=1)
            neighbor_indices = np.argsort(distances)[1:n_neighbors+1]
            
            # Create synthetic samples
            for _ in range(2):  # Create 2 samples per point
                # Randomly select a neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                # Get interpolation ratio
                ratio = np.random.beta(0.4, 0.4)  # Beta distribution for more diversity
                
                # Create new sample
                synthetic = ratio * class_X[idx] + (1 - ratio) * class_X[neighbor_idx]
                
                # Add small random noise
                noise = np.random.normal(0, noise_factor * 0.5 * feature_stds)
                synthetic += noise
                
                X_augmented.append(synthetic.reshape(1, -1))
                y_augmented.append([class_idx])
        
        # 3. Create boundary samples
        other_classes = [c for c in unique_classes if c != class_idx]
        for other_class in other_classes:
            other_X = X[class_indices[other_class]]
            
            # Find closest pairs between classes
            for idx in range(len(class_X)):
                distances = np.linalg.norm(other_X - class_X[idx], axis=1)
                closest_idx = np.argmin(distances)
                
                # Create boundary sample with careful interpolation
                ratio = np.random.beta(0.7, 0.7)  # Favor points closer to original class
                boundary = ratio * class_X[idx] + (1 - ratio) * other_X[closest_idx]
                
                X_augmented.append(boundary.reshape(1, -1))
                y_augmented.append([class_idx])
    
    # Convert lists to arrays
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.concatenate(y_augmented)
    
    # Shuffle the augmented dataset
    shuffle_idx = np.random.permutation(len(X_augmented))
    return X_augmented[shuffle_idx], y_augmented[shuffle_idx]

def preprocess_test_data(X_test):
    """Special preprocessing for test data to improve accuracy"""
    # Convert to tensor if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.FloatTensor(X_test)
    
    # Normalize each feature to [0,1] range
    X_min = X_test.min(dim=0)[0]
    X_max = X_test.max(dim=0)[0]
    X_normalized = (X_test - X_min) / (X_max - X_min + 1e-8)
    
    return X_normalized

def get_ensemble_predictions(model, X_test, n_augment=10):
    """Get predictions using test-time augmentation and ensemble"""
    model.eval()
    
    # Preprocess test data
    X_test = preprocess_test_data(X_test)
    
    all_probs = []
    with torch.no_grad():
        # Original prediction
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs)
        
        # Test-time augmentation
        for i in range(n_augment):
            # Add small random noise
            noise_scale = 0.02 * (1.0 - i/n_augment)  # Gradually decrease noise
            noise = torch.randn_like(X_test) * noise_scale
            aug_outputs = model(X_test + noise)
            aug_probs = torch.softmax(aug_outputs, dim=1)
            all_probs.append(aug_probs)
            
            # Also try slightly scaled versions
            scale = 1.0 + noise_scale
            aug_outputs = model(X_test * scale)
            aug_probs = torch.softmax(aug_outputs, dim=1)
            all_probs.append(aug_probs)
    
    # Average predictions
    all_probs = torch.stack(all_probs)
    avg_probs = torch.mean(all_probs, dim=0)
    
    # Get predictions with high confidence threshold
    confidence_threshold = 0.8
    max_probs, predictions = torch.max(avg_probs, dim=1)
    
    # For low confidence predictions, use mode from all augmented predictions
    low_confidence = max_probs < confidence_threshold
    if low_confidence.any():
        all_preds = torch.argmax(all_probs, dim=2)
        mode_preds = torch.mode(all_preds, dim=0).values
        predictions[low_confidence] = mode_preds[low_confidence]
    
    return predictions 