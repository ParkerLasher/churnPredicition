# Telco Customer Churn Prediction

This repository provides an end-to-end machine learning workflow for predicting customer churn in the telecommunications sector. The code demonstrates data preprocessing, model training with hyperparameter optimization, model evaluation, and feature interpretation using SHAP.

Note: This project is currently in the R&D phase. It is not optimized or integrated into a production environment and should be treated as a proof-of-concept or a starting point for further development.

## Project Overview

### Data Preprocessing (preprocess.py):
- Cleans the raw Telco Customer Churn dataset.
- Handles missing values, encodes categorical variables, and engineers new features.
- Outputs a fully processed dataset ready for modeling.
### Model Training (train.py):
- Splits the preprocessed data into training, validation, and test sets.
- Uses Optuna for Bayesian hyperparameter optimization on LightGBM and XGBoost models.
- Trains a Neural Network model using SciKeras.
- Combines the trained models into a Stacking Ensemble for improved performance.
- Identifies an optimal decision threshold based on the validation set.
- Saves trained models and parameters to the models directory.
### Model Evaluation (evaluate.py):
- Evaluates the trained Stacking Ensemble model on the test set.
- Prints out metrics such as classification report, confusion matrix, accuracy, and ROC AUC.
- Uses the previously computed optimal threshold for predictions.
### Feature Interpretation (shap_analysis.py):
- Computes SHAP values to explain model predictions and provide insight into feature importance.
- Generates a feature importance summary plot.
- Getting Started

## Prerequisites
Python 3.7+
Required Python packages listed in requirements.txt (if provided).
Data
Due to size constraints, the raw dataset from IBM cannot be uploaded directly to this repository.
Option 1: Download the dataset from the original source (IBM or provided link) and place it in the data directory.
Option 2: Use the preprocessed enhanced_data.csv if provided.

## Project Status

This repository reflects an R&D phase project. The code is functional but not optimized for production. It is intended to serve as a prototype or a reference implementation. The structure and logic may need refinement, testing, and modularization before being used in a production environment.

## Future Enhancements

### Production-Grade Pipelines:
- Integrate the steps into a continuous integration/continuous delivery (CI/CD) pipeline (e.g., using GitHub Actions, Jenkins, or Airflow).
- Automatically trigger data preprocessing, model retraining, and evaluation upon data updates or code changes.
### Model Serving and Deployment:
- Containerize the application with Docker.
- Deploy models as a REST API using frameworks like FastAPI or Flask.
- Integrate model monitoring and logging for drift detection and performance tracking.
### Scalability and Optimization:
- Implement feature stores for better data management.
- Leverage distributed training platforms or managed ML services for scaling up model training and evaluation.
### Modularization and Code Quality:
- Refactor the code into more maintainable modules.
- Increase test coverage with unit tests and integration tests.
### Documentation and Metadata:
- Add detailed docstrings and comments throughout the code.
- Utilize tools like MLflow or DVC for experiment tracking and artifact versioning.



