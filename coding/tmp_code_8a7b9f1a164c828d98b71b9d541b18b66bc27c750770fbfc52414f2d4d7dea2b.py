import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model, test data and predictions
model_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model/random_forest_h1n1_model.pkl'
test_data_path = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/data/test_set_features.csv'
pred_dir = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_predictions'
predictions_path = f'{pred_dir}/h1n1_predictions.csv'

rf_model = joblib.load(model_path)
test_data = pd.read_csv(test_data_path)
predictions = pd.read_csv(predictions_path)

# Get feature importances from the model
feature_importance = rf_model.feature_importances_
feature_names = rf_model.feature_names_in_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('Top 20 Features for H1N1 Vaccination Prediction')
plt.tight_layout()
plt.savefig(f'{pred_dir}/top_features_test_set.png')

# Merge predictions with test data for analysis
predictions_with_features = test_data.merge(predictions, on='respondent_id')

# Analyze top 10 most influential features for confident predictions
top_features = feature_importance_df.head(10)['Feature'].tolist()

# Get instances with high and low probabilities
high_prob = predictions_with_features[predictions_with_features['probability_vaccinated'] > 0.7]
low_prob = predictions_with_features[predictions_with_features['probability_vaccinated'] < 0.1]

# Function to analyze a feature across prediction groups
def analyze_feature(feature, high_prob_df, low_prob_df, all_df):
    if feature in all_df.columns:
        if all_df[feature].dtype in ['int64', 'float64']:
            # Numeric feature
            high_mean = high_prob_df[feature].mean()
            low_mean = low_prob_df[feature].mean()
            all_mean = all_df[feature].mean()
            
            return {
                'feature': feature,
                'high_prob_mean': high_mean,
                'low_prob_mean': low_mean,
                'all_mean': all_mean,
                'type': 'numeric'
            }
        else:
            # Categorical feature - count for each value
            high_counts = high_prob_df[feature].value_counts(normalize=True)
            low_counts = low_prob_df[feature].value_counts(normalize=True)
            all_counts = all_df[feature].value_counts(normalize=True)
            
            return {
                'feature': feature,
                'high_prob_distribution': high_counts.to_dict(),
                'low_prob_distribution': low_counts.to_dict(),
                'all_distribution': all_counts.to_dict(),
                'type': 'categorical'
            }
    else:
        # Feature not directly in dataframe (might be one-hot encoded)
        return {'feature': feature, 'type': 'one-hot', 'note': 'One-hot encoded feature not analyzed directly'}

# Analyze top features
feature_analysis = []
for feature in top_features:
    analysis = analyze_feature(feature, high_prob, low_prob, test_data)
    feature_analysis.append(analysis)

# Save feature analysis
feature_analysis_df = pd.DataFrame(feature_analysis)
feature_analysis_df.to_csv(f'{pred_dir}/feature_analysis.csv', index=False)

# Create summary report
with open(f'{pred_dir}/prediction_report.md', 'w') as f:
    f.write('# H1N1 Vaccination Prediction Report\n\n')
    
    # Summary of predictions
    f.write('## Prediction Summary\n')
    f.write(f'- Total test instances: {len(predictions)}\n')
    f.write(f'- Predicted vaccinated (class 1): {sum(predictions["predicted_class"])}\n')
    f.write(f'- Predicted not vaccinated (class 0): {len(predictions) - sum(predictions["predicted_class"])}\n')
    f.write(f'- Percentage predicted vaccinated: {(sum(predictions["predicted_class"]) / len(predictions)) * 100:.2f}%\n\n')
    
    # Distribution of prediction probabilities
    f.write('## Probability Distribution\n')
    f.write('The distribution of vaccination probability predictions is shown in the probability_distribution.png file.\n\n')
    
    # Top features
    f.write('## Top Features\n')
    f.write('The top 20 most important features for vaccination prediction are shown in the top_features_test_set.png file.\n\n')
    
    # Feature analysis for high vs low probability predictions
    f.write('## Feature Analysis for High vs Low Probability Predictions\n')
    for analysis in feature_analysis:
        f.write(f"### {analysis['feature']}\n")
        if analysis['type'] == 'numeric':
            f.write(f"- Mean value for high probability predictions: {analysis['high_prob_mean']:.4f}\n")
            f.write(f"- Mean value for low probability predictions: {analysis['low_prob_mean']:.4f}\n")
            f.write(f"- Mean value across all predictions: {analysis['all_mean']:.4f}\n\n")
        elif analysis['type'] == 'categorical':
            f.write("- Distribution for high probability predictions:\n")
            for category, value in analysis['high_prob_distribution'].items():
                f.write(f"  - {category}: {value:.4f}\n")
            f.write("- Distribution for low probability predictions:\n")
            for category, value in analysis['low_prob_distribution'].items():
                f.write(f"  - {category}: {value:.4f}\n")
            f.write("\n")
        else:
            f.write(f"- Note: {analysis['note']}\n\n")
    
    # Conclusion
    f.write('## Conclusion\n')
    f.write('The Random Forest model predicted that about 10.52% of individuals in the test set would get the H1N1 vaccine. This is lower than the training set percentage (21.25%), which might indicate differences in the test population or potential model improvements needed.\n\n')
    
    # Files
    f.write('## Files\n')
    f.write('- h1n1_predictions.csv: Full prediction results\n')
    f.write('- sample_predictions.csv: Sample of 100 predictions\n')
    f.write('- probability_distribution.png: Histogram of prediction probabilities\n')
    f.write('- top_features_test_set.png: Top 20 features by importance\n')
    f.write('- feature_analysis.csv: Analysis of feature values for high and low probability predictions\n')

# List all files in predictions directory
print("Prediction report and analysis completed.")
print("\nFiles in the predictions directory:")
for file in os.listdir(pred_dir):
    print(f"- {file}")

print(f"\nPrediction report saved to: {pred_dir}/prediction_report.md")