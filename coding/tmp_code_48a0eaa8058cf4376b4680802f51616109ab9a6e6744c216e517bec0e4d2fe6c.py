import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model/random_forest_h1n1_model.pkl'
test_data_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/test_set_features.csv'

# Check if files exist
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Test data file exists: {os.path.exists(test_data_path)}")

# Create directory for prediction results
pred_dir = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_predictions'
os.makedirs(pred_dir, exist_ok=True)

# Load the model
rf_model = joblib.load(model_path)
print("Model loaded successfully")

# Load test data
test_data = pd.read_csv(test_data_path)
print(f"Test data shape: {test_data.shape}")
print("First few rows of test data:")
print(test_data.head())

# Check for columns in test data
print("\nTest data columns:")
print(test_data.columns.tolist())