import os
import pandas as pd
import shap
from joblib import load
import matplotlib.pyplot as plt

# Load enhanced data
data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'enhanced_data.csv')
data = pd.read_csv(data_path)

# Prepare features
X = data.drop(['Churn_Yes'], axis=1)

# Load LightGBM model
lgbm_model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'lgbm_model.joblib')
best_lgbm_model = load(lgbm_model_path)

# Extract the classifier from the pipeline
classifier = best_lgbm_model.named_steps['classifier']

# Apply ADASYN to the entire dataset to match the training distribution
adasyn = best_lgbm_model.named_steps['adasyn']
X_resampled, y_resampled = adasyn.fit_resample(X, data['Churn_Yes'])

# Compute SHAP values
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_resampled)

# Plot summary plot for feature importance
shap.summary_plot(shap_values, X_resampled, plot_type="bar")
plt.show()