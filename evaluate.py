import os
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from utils import load_enhanced_data

# Load enhanced data
data = load_enhanced_data()

# Prepare features and target
X = data.drop(['Churn_Yes'], axis=1)
y = data['Churn_Yes']

# Split the data into train and test sets (consistent with train.py)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Load the stacking model
stack_model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'stack_model.joblib')
stack_model = load(stack_model_path)

# Load the optimal threshold
threshold_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'optimal_threshold.npy')
optimal_threshold = np.load(threshold_path)
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Get predicted probabilities on the test set
test_preds_proba = stack_model.predict_proba(X_test)[:, 1]

# Apply the optimal threshold
test_preds = np.where(test_preds_proba >= optimal_threshold, 1, 0)

# Evaluate the model
print("Classification Report on Test Set:")
print(classification_report(y_test, test_preds, target_names=['No Churn', 'Churn']))

print("Confusion Matrix on Test Set:")
conf_matrix = confusion_matrix(y_test, test_preds)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual No Churn', 'Actual Churn'],
                              columns=['Predicted No Churn', 'Predicted Churn'])
print(conf_matrix_df)

print("Accuracy Score on Test Set:")
accuracy = accuracy_score(y_test, test_preds)
print(f"{accuracy:.4f}")

print("ROC AUC Score on Test Set:")
roc_auc = roc_auc_score(y_test, test_preds_proba)
print(f"{roc_auc:.4f}")