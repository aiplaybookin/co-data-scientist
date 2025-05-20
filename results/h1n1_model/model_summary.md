# H1N1 Vaccine Prediction Model Summary

## Dataset Information
- Dataset: H1N1 Vaccine Prediction
- Target: h1n1_vaccine (binary classification: 0 - not vaccinated, 1 - vaccinated)
- Features: 36 input variables including demographic information, behaviors, and opinions
- Missing values: Several columns had missing values, which were handled during preprocessing

## Preprocessing Steps
1. Filled missing numeric values with median
2. One-hot encoded categorical variables
3. Dropped the respondent_id column as it is not a predictor
4. Split data into 80% training and 20% testing sets

## Model Information
- Algorithm: Random Forest Classifier
- Parameters:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
  - min_samples_leaf: 2

## Model Performance
```
Accuracy: 0.8325
Precision: 0.7335
Recall: 0.3322
F1 Score: 0.4572
ROC-AUC: 0.8225
PR-AUC: 0.6141

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.97      0.90      4207
           1       0.73      0.33      0.46      1135

    accuracy                           0.83      5342
   macro avg       0.79      0.65      0.68      5342
weighted avg       0.82      0.83      0.81      5342
```

## Top Features
The most important features for predicting H1N1 vaccination status are shown in the feature_importance.png image.

## Visualizations
- **ROC Curve**: Shows the trade-off between true positive rate and false positive rate (roc_curve.png)
- **Precision-Recall Curve**: Shows the trade-off between precision and recall (precision_recall_curve.png)
- **Confusion Matrix**: Shows the counts of true positives, false positives, true negatives, and false negatives (confusion_matrix.png)

## Conclusion
The Random Forest model achieved good overall accuracy, but with moderate recall, indicating that the model is better at identifying people who did not get vaccinated than those who did. The ROC-AUC score shows good discriminative ability. Further tuning could potentially improve the recall.

## Files
- random_forest_h1n1_model.pkl: The trained model
- feature_importance.png: Bar chart of feature importance
- roc_curve.png: ROC curve visualization
- precision_recall_curve.png: Precision-Recall curve visualization
- confusion_matrix.png: Confusion matrix visualization
- classification_report.csv: Detailed classification metrics
- model_evaluation.txt: Summary of model performance metrics
