import pandas as pd
import os

# Check file path and load data
file_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/h1n1_dataset.csv'
print(f"Does file exist: {os.path.exists(file_path)}")

# Read the data
data = pd.read_csv(file_path)

# Display basic information
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nBasic info:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())

# Check target variable
print("\nTarget variable distribution:")
print(data['h1n1_vaccine'].value_counts())
print(data['h1n1_vaccine'].value_counts(normalize=True))