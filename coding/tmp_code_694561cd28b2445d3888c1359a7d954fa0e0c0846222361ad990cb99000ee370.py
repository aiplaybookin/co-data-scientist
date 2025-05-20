import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model/random_forest_h1n1_model.pkl'
rf_model = joblib.load(model_path)

# Get feature names used by the model
model_features = rf_model.feature_names_in_
print(f"Number of features used by the model: {len(model_features)}")
print("First 10 model features:")
print(model_features[:10])

# Load original training data to get feature transformation info
original_data_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/h1n1_dataset.csv'
original_data = pd.read_csv(original_data_path)

# Load test data
test_data_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/test_set_features.csv'
test_data = pd.read_csv(test_data_path)

# We need to apply the same preprocessing to test data that we did to training data
# 1. Handle missing numeric values
numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())

# 2. Get categorical columns from test data
categorical_columns = test_data.select_dtypes(include=['object']).columns
print("\nCategorical columns in test data:")
print(list(categorical_columns))

# 3. Apply one-hot encoding
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# 4. Drop respondent_id
test_encoded = test_encoded.drop('respondent_id', axis=1)

# Check for any remaining missing values
remaining_missing = test_encoded.isnull().sum().sum()
print(f"\nRemaining missing values in test data after preprocessing: {remaining_missing}")

# Check for feature mismatch
test_features = set(test_encoded.columns)
model_features_set = set(model_features)

print("\nFeature comparison:")
print(f"Features in test data: {len(test_features)}")
print(f"Features in model: {len(model_features_set)}")

# Features in test but not in model
print("\nFeatures in test data but not in model:")
print(test_features - model_features_set)

# Features in model but not in test
print("\nFeatures in model but not in test data:")
print(model_features_set - test_features)