"""
Logistic Regression Training Script for Iris Dataset
This script trains a logistic regression model on the iris dataset
and saves the trained model to the models directory.
"""

import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime


def load_data():
    """Load iris dataset and prepare train/test split"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, iris


def preprocess_data(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """Train logistic regression model"""
    model = LogisticRegression(
        max_iter=200,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics, y_pred


def save_model(model, scaler, iris, metrics):
    """Save trained model and scaler"""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'model_type': 'Logistic Regression',
        'dataset': 'Iris',
        'features': iris.feature_names,
        'classes': iris.target_names.tolist(),
        'metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'input_features': 4,
        'output_classes': 3
    }
    
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Iris Logistic Regression Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading iris dataset...")
    X_train, X_test, y_train, y_test, iris = load_data()
    print(f"✓ Training set size: {X_train.shape[0]}")
    print(f"✓ Test set size: {X_test.shape[0]}")
    print(f"✓ Features: {X_train.shape[1]}")
    print(f"✓ Classes: {len(np.unique(y_train))}")
    
    # Preprocess data
    print("\n[2/5] Preprocessing data (scaling)...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    print("✓ Data scaled using StandardScaler")
    
    # Train model
    print("\n[3/5] Training logistic regression model...")
    model = train_model(X_train_scaled, y_train)
    print("✓ Model training completed")
    
    # Evaluate model
    print("\n[4/5] Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test_scaled, y_test)
    
    print("\nModel Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save model
    print("\n[5/5] Saving model artifacts...")
    save_model(model, scaler, iris, metrics)
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
