from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import json
import torch
from sklearn.metrics import confusion_matrix
from datetime import datetime

def generate_test_datasets():
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Save test data
    torch.save({
        'X_test': X_test_tensor,
        'y_test': y_test_tensor
    }, 'pytorch/neural_networks/iris_dataset/data/test_dataset.pt')
    
    # Generate sample historical predictions for drift testing
    # Using more balanced probabilities that better reflect typical Iris classification
    sample_predictions = np.random.choice(3, size=len(y_test), p=[0.33, 0.33, 0.34])
    
    # Make predictions more aligned with actual labels to reduce drift
    for i in range(len(sample_predictions)):
        # 90% chance to predict the correct label
        if np.random.random() < 0.9:
            sample_predictions[i] = y_test[i]
            
    conf_matrix = confusion_matrix(y_test, sample_predictions).tolist()
    
    historical_data = {
        'predictions': sample_predictions.tolist(),
        'confusion_matrix': conf_matrix,
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d'),
            'model_version': '1.0.0',
            'dataset_size': len(y_test)
        }
    }
    
    # Save historical predictions
    with open('pytorch/neural_networks/iris_dataset/data/historical_predictions.json', 'w') as f:
        json.dump(historical_data, f, indent=2)

if __name__ == "__main__":
    generate_test_datasets() 