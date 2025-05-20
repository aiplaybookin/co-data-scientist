import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the data
file_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/h1n1_dataset.csv'
data = pd.read_csv(file_path)

# Check missing values
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
missing_df = pd.DataFrame({'missing_count': missing_values, 'missing_percentage': missing_percentage})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percentage', ascending=False)

print("Missing values by column:")
print(missing_df)

# Create a directory to save the results
results_dir = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model'
os.makedirs(results_dir, exist_ok=True)

# Save missing values information
plt.figure(figsize=(12, 8))
plt.bar(missing_df.index, missing_df['missing_percentage'])
plt.xticks(rotation=90)
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig(f'{results_dir}/missing_values.png')

# Check for categorical columns and convert them to one-hot encoding
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nCategorical columns:", list(categorical_columns))

# Prepare the data - separate features from target
X = data.drop('h1n1_vaccine', axis=1)
y = data['h1n1_vaccine']

# Handle missing values (start with numeric columns)
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Convert categorical features to one-hot encoding
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print("\nShape after one-hot encoding:", X_encoded.shape)

# Drop respondent_id as it's not a useful feature
X_encoded = X_encoded.drop('respondent_id', axis=1)

# Final check for missing values
remaining_missing = X_encoded.isnull().sum().sum()
print(f"\nRemaining missing values after preprocessing: {remaining_missing}")

# Save preprocessed data info
with open(f'{results_dir}/data_preprocessing_summary.txt', 'w') as f:
    f.write(f"Original dataset shape: {data.shape}\n")
    f.write(f"Processed features shape: {X_encoded.shape}\n")
    f.write(f"Target distribution:\n{y.value_counts().to_string()}\n")
    f.write(f"Target percentage:\n{y.value_counts(normalize=True).to_string()}")

print("Preprocessing complete and summary saved.")