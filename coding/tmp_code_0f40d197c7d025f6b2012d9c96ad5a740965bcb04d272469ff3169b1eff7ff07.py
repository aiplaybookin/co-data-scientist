import pandas as pd
import os

# Create a summary document
results_dir = '/Users/vikash/Documents/2025/gsk/demo-repo/co-data-scientist/results/h1n1_model'

with open(f'{results_dir}/model_summary.md', 'w') as f:
    f.write('# H1N1 Vaccine Prediction Model Summary\n\n')
    
    # Dataset information
    f.write('## Dataset Information\n')
    f.write('- Dataset: H1N1 Vaccine Prediction\n')
    f.write('- Target: h1n1_vaccine (binary classification: 0 - not vaccinated, 1 - vaccinated)\n')
    f.write('- Features: 36 input variables including demographic information, behaviors, and opinions\n')
    f.write('- Missing values: Several columns had missing values, which were handled during preprocessing\n\n')
    
    # Preprocessing steps
    f.write('## Preprocessing Steps\n')
    f.write('1. Filled missing numeric values with median\n')
    f.write('2. One-hot encoded categorical variables\n')
    f.write('3. Dropped the respondent_id column as it is not a predictor\n')
    f.write('4. Split data into 80% training and 20% testing sets\n\n')
    
    # Model information
    f.write('## Model Information\n')
    f.write('- Algorithm: Random Forest Classifier\n')
    f.write('- Parameters:\n')
    f.write('  - n_estimators: 100\n')
    f.write('  - max_depth: 10\n')
    f.write('  - min_samples_split: 5\n')
    f.write('  - min_samples_leaf: 2\n\n')
    
    # Model performance
    f.write('## Model Performance\n')
    
    # Read the evaluation metrics
    with open(f'{results_dir}/model_evaluation.txt', 'r') as eval_file:
        metrics = eval_file.read()
    
    f.write('```\n')
    f.write(metrics)
    f.write('```\n\n')
    
    # Feature importance
    f.write('## Top Features\n')
    f.write('The most important features for predicting H1N1 vaccination status are shown in the feature_importance.png image.\n\n')
    
    # Visualizations
    f.write('## Visualizations\n')
    f.write('- **ROC Curve**: Shows the trade-off between true positive rate and false positive rate (roc_curve.png)\n')
    f.write('- **Precision-Recall Curve**: Shows the trade-off between precision and recall (precision_recall_curve.png)\n')
    f.write('- **Confusion Matrix**: Shows the counts of true positives, false positives, true negatives, and false negatives (confusion_matrix.png)\n\n')
    
    # Conclusion
    f.write('## Conclusion\n')
    f.write('The Random Forest model achieved good overall accuracy, but with moderate recall, indicating that the model is better at identifying people who did not get vaccinated than those who did. The ROC-AUC score shows good discriminative ability. Further tuning could potentially improve the recall.\n\n')
    
    # Files
    f.write('## Files\n')
    f.write('- random_forest_h1n1_model.pkl: The trained model\n')
    f.write('- feature_importance.png: Bar chart of feature importance\n')
    f.write('- roc_curve.png: ROC curve visualization\n')
    f.write('- precision_recall_curve.png: Precision-Recall curve visualization\n')
    f.write('- confusion_matrix.png: Confusion matrix visualization\n')
    f.write('- classification_report.csv: Detailed classification metrics\n')
    f.write('- model_evaluation.txt: Summary of model performance metrics\n')

print(f"Model summary created at: {results_dir}/model_summary.md")

# List all files in the results directory
print("\nFiles in the results directory:")
for file in os.listdir(results_dir):
    print(f"- {file}")