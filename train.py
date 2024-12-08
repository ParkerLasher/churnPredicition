import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from joblib import dump
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
import optuna
from utils import load_enhanced_data
import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Import the neural network model function from model_definitions.py
from model_definitions import create_nn_model

# =======================
# Setup Paths and Directories
# =======================
models_dir = os.path.join(os.path.dirname(os.getcwd()), 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load enhanced data
data = load_enhanced_data()

# Prepare features and target
X = data.drop(['Churn_Yes'], axis=1)
y = data['Churn_Yes']

# Split the data into train, val, and test sets (60% train, 20% val, 20% test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full)

input_dim = X_train.shape[1]

# Handle class imbalance using ADASYN
adasyn = ADASYN(random_state=42)

# =======================
# LightGBM with Bayesian Optimization
# =======================
def objective_lgbm(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42
    }
    model = ImbPipeline(steps=[
        ('adasyn', adasyn),
        ('classifier', LGBMClassifier(**param))
    ])
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=50)
best_params_lgbm = study_lgbm.best_trial.params
print(f"Best LightGBM parameters: {best_params_lgbm}")

best_lgbm_model = ImbPipeline(steps=[
    ('adasyn', adasyn),
    ('classifier', LGBMClassifier(**best_params_lgbm))
])
best_lgbm_model.fit(X_train_full, y_train_full)

lgbm_model_path = os.path.join(models_dir, 'lgbm_model.joblib')
dump(best_lgbm_model, lgbm_model_path)
print(f"Best LightGBM model trained and saved to: {lgbm_model_path}")

# =======================
# XGBoost with Bayesian Optimization
# =======================
def objective_xgb(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = ImbPipeline(steps=[
        ('adasyn', adasyn),
        ('classifier', XGBClassifier(**param))
    ])
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    return auc

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
best_params_xgb = study_xgb.best_trial.params
print(f"Best XGBoost parameters: {best_params_xgb}")

best_xgb_model = ImbPipeline(steps=[
    ('adasyn', adasyn),
    ('classifier', XGBClassifier(**best_params_xgb))
])
best_xgb_model.fit(X_train_full, y_train_full)

xgb_model_path = os.path.join(models_dir, 'xgb_model.joblib')
dump(best_xgb_model, xgb_model_path)
print(f"Best XGBoost model trained and saved to: {xgb_model_path}")

# =======================
# Neural Network with SciKeras
# =======================
# Using create_nn_model from model_definitions.py
nn_model = ImbPipeline(steps=[
    ('adasyn', adasyn),
    ('classifier', KerasClassifier(
        model=create_nn_model,
        model__input_dim=input_dim,
        epochs=50,
        batch_size=32,
        verbose=0,
        random_state=42
    ))
])

nn_model.fit(X_train_full, y_train_full)
nn_model_path = os.path.join(models_dir, 'nn_model.joblib')
dump(nn_model, nn_model_path)
print(f"Neural Network model trained and saved to: {nn_model_path}")

# =======================
# Stacking Ensemble
# =======================
estimators = [
    ('lgbm', best_lgbm_model),
    ('xgb', best_xgb_model),
    ('nn', nn_model)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stack_model.fit(X_train_full, y_train_full)

# Compute optimal threshold based on the validation set using stack_model
val_preds_proba = stack_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_preds_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold based on F1-score: {optimal_threshold:.4f}")

# Refit stack_model on train+val for final model
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)
stack_model.fit(X_train_val, y_train_val)

stack_model_path = os.path.join(models_dir, 'stack_model.joblib')
dump(stack_model, stack_model_path)
print(f"Stacking model trained and saved to: {stack_model_path}")

threshold_path = os.path.join(models_dir, 'optimal_threshold.npy')
np.save(threshold_path, optimal_threshold)
print(f"Optimal threshold saved to: {threshold_path}")

print("Training completed successfully and all models saved.")