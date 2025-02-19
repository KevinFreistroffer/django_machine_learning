import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nn_lightning import IrisClassifier
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

def validate_model():
    # Load test dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data/test_dataset.pt')
    test_data = torch.load(data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints/model.ckpt')
    model = IrisClassifier.load_from_checkpoint(model_path)
    model.eval()
    
    # Collect predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average='weighted'
    )
    
    # Define thresholds
    ACCURACY_THRESHOLD = 0.90
    PRECISION_THRESHOLD = 0.85
    RECALL_THRESHOLD = 0.85
    F1_THRESHOLD = 0.85
    
    # Validate metrics
    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy {accuracy:.2f} below threshold {ACCURACY_THRESHOLD}"
    assert precision >= PRECISION_THRESHOLD, f"Precision {precision:.2f} below threshold {PRECISION_THRESHOLD}"
    assert recall >= RECALL_THRESHOLD, f"Recall {recall:.2f} below threshold {RECALL_THRESHOLD}"
    assert f1 >= F1_THRESHOLD, f"F1 {f1:.2f} below threshold {F1_THRESHOLD}"
    
    print(f"Model validation passed!")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    validate_model()