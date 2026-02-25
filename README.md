# Iris Logistic Regression MLOps Project

A machine learning project demonstrating MLOps best practices using a Logistic Regression model trained on the Iris dataset.

## Project Structure

```
mlops-unit1/
├── data/                      # Data directory (train, test, raw data)
├── src/                       # Source code directory
│   ├── train.py              # Model training script
│   └── predict.py            # Model prediction script
├── models/                    # Trained model artifacts
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Overview

This project implements a simple yet production-ready machine learning pipeline for binary/multiclass classification using Logistic Regression on the famous Iris dataset.

### Key Features

- **Modular Design**: Separate scripts for training and prediction
- **Data Preprocessing**: StandardScaler normalization
- **Model Persistence**: Saved model artifacts with metadata
- **Performance Metrics**: Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- **MLOps Friendly**: Clear folder structure and documentation

## Dataset

**Iris Dataset**
- Samples: 150 (120 train, 30 test)
- Features: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- Classes: 3 (setosa, versicolor, virginica)
- Train-Test Split: 80-20 with stratification

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaegar03/ML_OOPS_3RD-EXERCISE.git
cd ML_OOPS_3RD-EXERCISE/mlops-unit1
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the logistic regression model:

```bash
python src/train.py
```

**Output:**
- Trained model: `models/logistic_regression_model.pkl`
- Feature scaler: `models/scaler.pkl`
- Model metadata: `models/model_metadata.json`

### Making Predictions

Use the trained model to make predictions on new data:

```bash
python src/predict.py
```

## Model Performance

The trained model achieves the following performance metrics on the test set:

- **Accuracy**: High accuracy across all iris classes
- **Precision**: Weighted average precision
- **Recall**: Excellent recall on all classes
- **F1-Score**: Balanced performance metric

(Specific metrics are logged when running the training script)

## Model Architecture

**Algorithm**: Logistic Regression
- Solver: lbfgs
- Max Iterations: 200
- Multi-class: multinomial
- Random State: 42 (for reproducibility)

## Files Description

### `src/train.py`
Main training pipeline that:
1. Loads the iris dataset
2. Performs train-test split (80-20)
3. Scales features using StandardScaler
4. Trains the logistic regression model
5. Evaluates performance metrics
6. Saves model artifacts

### `src/predict.py`
Prediction script that:
1. Loads trained model and scaler
2. Makes predictions on new samples
3. Provides confidence scores
4. Displays results with class names

## Requirements

- Python 3.8+
- scikit-learn 1.3.2+
- numpy 1.24.3+
- pandas 2.0.3+
- matplotlib 3.7.2+
- seaborn 0.12.2+

## MLOps Best Practices Demonstrated

1. ✓ Modular code structure
2. ✓ Separate concerns (training, prediction)
3. ✓ Model versioning with metadata
4. ✓ Feature scaling and preprocessing
5. ✓ Reproducible results (random_state)
6. ✓ Comprehensive logging and metrics
7. ✓ Model persistence and artifacts
8. ✓ Clear documentation

## Future Enhancements

- Model evaluation scripts with visualization
- Cross-validation and hyperparameter tuning
- Model versioning system (DVC)
- Continuous integration/deployment (CI/CD)
- Model registry and deployment
- Unit tests for model components
- Docker containerization

## Author

**Username**: JAGAR03  
**Email**: kunutk03@gmail.com

## License

MIT License

## References

- [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [Logistic Regression - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MLOps Best Practices](https://ml-ops.systems/)
