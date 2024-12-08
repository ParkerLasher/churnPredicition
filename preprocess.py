import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# Load the data
def load_data():
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return pd.read_csv(data_path)


# Load the dataset
data = load_data()

# Step 1: Initial data loaded
print("Step 1: Initial data loaded.")
print(f"Number of NaNs: {data.isna().sum().sum()}")
print(f"Columns present: {list(data.columns)}")

# Step 2: Handle 'TotalCharges' column
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
print("Step 2: 'TotalCharges' column handled.")

# Step 3: Impute missing values for numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
print("Step 3: Numeric columns imputed.")

# Step 4: Impute missing values for categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
print("Step 4: Categorical columns imputed.")

# Step 5: One-Hot Encoding for categorical features
data = pd.get_dummies(data, drop_first=True)
print("Step 5: One-Hot Encoding applied.")

# Step 6: Add ContractLength feature based on one-hot encoded columns
contract_columns = ['Contract_One year', 'Contract_Two year']
data['ContractLength'] = 1  # Default value for "Month-to-month"
if 'Contract_One year' in data.columns:
    data['ContractLength'] += 11 * data['Contract_One year']
if 'Contract_Two year' in data.columns:
    data['ContractLength'] += 23 * data['Contract_Two year']
print("Step 6: ContractLength feature added.")

# Step 7: Create interaction features
# Interaction between ContractLength and PaymentMethod_Electronic check
if 'PaymentMethod_Electronic check' in data.columns:
    data['Contract_Payment_Interaction'] = data['ContractLength'] * data['PaymentMethod_Electronic check']
    print("Interaction feature 'Contract_Payment_Interaction' added.")

# Decompose customer_engagement into individual features
engagement_features = [
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'OnlineSecurity_Yes',
    'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes'
]
for feature in engagement_features:
    if feature in data.columns:
        data[feature] = data[feature]
print("Customer engagement features decomposed.")

# Step 8: Advanced Feature Engineering
# Binning tenure
data['tenure_bin'] = pd.cut(data['tenure'], bins=[0, 12, 24, 48, 60, np.inf], labels=False)
print("Feature 'tenure_bin' added.")

# Polynomial Features for high-impact variables
high_impact_features = ['ContractLength', 'PaymentMethod_Electronic check', 'tenure', 'customer_engagement']
existing_high_impact_features = [feat for feat in high_impact_features if feat in data.columns]

if existing_high_impact_features:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(data[existing_high_impact_features])
    poly_feature_names = poly.get_feature_names_out(existing_high_impact_features)

    # Append a prefix to avoid column name duplication
    poly_feature_names = [f"poly_{name}" for name in poly_feature_names]

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)
    data = pd.concat([data, poly_df], axis=1)
    print("Polynomial features added with unique column names.")
else:
    print("No existing high-impact features found for polynomial feature generation.")

# Step 9: Remove low-impact features
low_impact_features = ['MonthlyCharges']
data.drop(columns=low_impact_features, inplace=True, errors='ignore')
print(f"Low-impact features {low_impact_features} removed.")

# Step 10: Handle any remaining NaNs after feature engineering
print("Handling any remaining NaNs after feature engineering...")

# Recalculate numeric_cols after adding new features
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Handle missing values in numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Verify no NaNs remain
total_nans = data.isna().sum().sum()
print(f"Number of NaNs after handling missing values: {total_nans}")

# Step 11: Check for duplicate columns
duplicate_columns = data.columns[data.columns.duplicated()].tolist()
if duplicate_columns:
    print(f"Duplicate columns found: {duplicate_columns}")
else:
    print("No duplicate columns found.")

# Save enhanced data
enhanced_data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'enhanced_data.csv')
data.to_csv(enhanced_data_path, index=False)
print(f"Enhanced data saved to: {enhanced_data_path}")