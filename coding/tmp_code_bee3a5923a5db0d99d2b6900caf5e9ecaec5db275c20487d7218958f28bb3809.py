import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Load the model and test data
model_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model/random_forest_h1n1_model.pkl'
test_data_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/test_set_features.csv'
rf_model = joblib.load(model_path)
test_data = pd.read_csv(test_data_path)

# Create directory for prediction results
pred_dir = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_predictions'
os.makedirs(pred_dir, exist_ok=True)

# Preprocess test data
# 1. Handle missing numeric values
numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())

# 2. Apply one-hot encoding
categorical_columns = test_data.select_dtypes(include=['object']).columns
test_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

# 3. Store respondent_id before dropping it
respondent_ids = test_data['respondent_id'].copy()

# 4. Drop respondent_id for prediction
test_encoded = test_encoded.drop('respondent_id', axis=1)

# Make predictions
y_pred = rf_model.predict(test_encoded)
y_prob = rf_model.predict_proba(test_encoded)[:, 1]  # Probability of class 1

# Create prediction dataframe
predictions_df = pd.DataFrame({
    'respondent_id': respondent_ids,
    'predicted_class': y_pred,
    'probability_vaccinated': y_prob
})

# Save predictions to CSV
predictions_path = f'{pred_dir}/h1n1_predictions.csv'
predictions_df.to_csv(predictions_path, index=False)

# Summary statistics of predictions
print("Prediction Summary:")
print(f"Total test instances: {len(predictions_df)}")
print(f"Predicted vaccinated (class 1): {sum(y_pred)}")
print(f"Predicted not vaccinated (class 0): {len(y_pred) - sum(y_pred)}")
print(f"Percentage predicted vaccinated: {(sum(y_pred) / len(y_pred)) * 100:.2f}%")

# Histogram of prediction probabilities
plt.figure(figsize=(10, 6))
plt.hist(y_prob, bins=20, edgecolor='black')
plt.title('Distribution of Vaccination Probability Predictions')
plt.xlabel('Probability of Vaccination')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.savefig(f'{pred_dir}/probability_distribution.png')

# Look at top 10 most confident predictions for each class
top_vaccinated = predictions_df[predictions_df['predicted_class'] == 1].sort_values('probability_vaccinated', ascending=False).head(10)
top_not_vaccinated = predictions_df[predictions_df['predicted_class'] == 0].sort_values('probability_vaccinated', ascending=True).head(10)

print("\nTop 10 most confident vaccinated predictions:")
print(top_vaccinated)

print("\nTop 10 most confident not vaccinated predictions:")
print(top_not_vaccinated)

# Save a sample of predictions for inspection
sample_size = min(100, len(predictions_df))
sample_predictions = predictions_df.sample(sample_size, random_state=42)
sample_predictions.to_csv(f'{pred_dir}/sample_predictions.csv', index=False)

print(f"\nPredictions saved to: {predictions_path}")
print(f"Sample predictions saved to: {pred_dir}/sample_predictions.csv")
print(f"Probability distribution plot saved to: {pred_dir}/probability_distribution.png")