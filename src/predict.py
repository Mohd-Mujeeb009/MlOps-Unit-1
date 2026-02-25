"""
Model prediction script
This script loads the trained model and makes predictions on new data.
"""

import pickle
import os
import json
import numpy as np
from sklearn.datasets import load_iris


def load_model_artifacts():
    """Load the trained model and scaler"""
    models_dir = 'models'
    
    # Load model
    with open(os.path.join(models_dir, 'logistic_regression_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def predict(features):
    """
    Make prediction on new data
    
    Args:
        features: numpy array of shape (n_samples, 4)
    
    Returns:
        predictions: predicted class labels
        probabilities: prediction probabilities
    """
    model, scaler, metadata = load_model_artifacts()
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return predictions, probabilities, metadata


def main():
    """Example: predict on test samples from iris dataset"""
    print("=" * 60)
    print("Iris Logistic Regression - Model Prediction")
    print("=" * 60)
    
    # Load iris dataset for test samples
    iris = load_iris()
    X = iris.data[:5]  # First 5 samples
    y_true = iris.target[:5]
    
    print(f"\nMaking predictions on {len(X)} samples...")
    
    predictions, probabilities, metadata = predict(X)
    
    print(f"\nClass names: {metadata['classes']}")
    print("\nPredictions:")
    print("-" * 60)
    
    for i, (pred, probs, true) in enumerate(zip(predictions, probabilities, y_true)):
        pred_name = metadata['classes'][pred]
        true_name = metadata['classes'][true]
        confidence = np.max(probs)
        
        print(f"Sample {i+1}:")
        print(f"  Predicted: {pred_name} (confidence: {confidence:.4f})")
        print(f"  Actual:    {true_name}")
        print(f"  Match: {'✓' if pred == true else '✗'}")
        print()


if __name__ == "__main__":
    main()
