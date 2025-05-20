# H1N1 Vaccination Prediction Report

## Prediction Summary
- Total test instances: 2671
- Predicted vaccinated (class 1): 281
- Predicted not vaccinated (class 0): 2390
- Percentage predicted vaccinated: 10.52%

## Probability Distribution
The distribution of vaccination probability predictions is shown in the probability_distribution.png file.

## Top Features
The top 20 most important features for vaccination prediction are shown in the top_features_test_set.png file.

## Feature Analysis for High vs Low Probability Predictions
### doctor_recc_h1n1
- Mean value for high probability predictions: 0.7234
- Mean value for low probability predictions: 0.0013
- Mean value across all predictions: 0.2263

### opinion_h1n1_risk
- Mean value for high probability predictions: 4.3404
- Mean value for low probability predictions: 1.4439
- Mean value across all predictions: 2.3327

### opinion_h1n1_vacc_effective
- Mean value for high probability predictions: 4.9362
- Mean value for low probability predictions: 3.2419
- Mean value across all predictions: 3.8494

### opinion_seas_risk
- Mean value for high probability predictions: 4.3191
- Mean value for low probability predictions: 1.6586
- Mean value across all predictions: 2.6989

### doctor_recc_seasonal
- Mean value for high probability predictions: 0.7234
- Mean value for low probability predictions: 0.0781
- Mean value across all predictions: 0.3313

### health_worker
- Mean value for high probability predictions: 0.7872
- Mean value for low probability predictions: 0.0136
- Mean value across all predictions: 0.1041

### opinion_seas_vacc_effective
- Mean value for high probability predictions: 4.9787
- Mean value for low probability predictions: 3.3621
- Mean value across all predictions: 4.0190

### employment_industry_fcxhlnwr
- Note: One-hot encoded feature not analyzed directly

### h1n1_knowledge
- Mean value for high probability predictions: 1.8511
- Mean value for low probability predictions: 1.1309
- Mean value across all predictions: 1.2588

### h1n1_concern
- Mean value for high probability predictions: 2.0426
- Mean value for low probability predictions: 1.1730
- Mean value across all predictions: 1.6360

## Conclusion
The Random Forest model predicted that about 10.52% of individuals in the test set would get the H1N1 vaccine. This is lower than the training set percentage (21.25%), which might indicate differences in the test population or potential model improvements needed.

## Files
- h1n1_predictions.csv: Full prediction results
- sample_predictions.csv: Sample of 100 predictions
- probability_distribution.png: Histogram of prediction probabilities
- top_features_test_set.png: Top 20 features by importance
- feature_analysis.csv: Analysis of feature values for high and low probability predictions
