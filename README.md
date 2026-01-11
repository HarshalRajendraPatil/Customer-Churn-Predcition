# Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Udemy Course](https://img.shields.io/badge/Based%20on-Udemy%20ML%20Course-orange)](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery)

## Overview

This project focuses on predicting whether a telecom customer will churn (leave the service) in the next month, enabling proactive retention strategies. It uses customer behavior, usage patterns, and subscription information as inputs to output churn probabilities (0-1). Success is measured by model accuracy, recall, precision, and expected monetary retention uplift.

The project is an extension of the "Complete Machine Learning and Data Science: Zero to Mastery" Udemy course (completed up to Section 13, including capstone projects). It demonstrates end-to-end ML workflow: data loading, EDA, preprocessing, model training with hyperparameter tuning (GridSearchCV), evaluation, and business metric calculation.

Key features:
- Dataset: Telco Customer Churn (~7,000 records from Kaggle/IBM).
- Models: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest Classifier.
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- Hyperparameter Tuning: Extensive GridSearchCV for optimal model performance.
- Business Insight: Calculates potential revenue uplift from retaining predicted churners.

## Table of Contents

- [Dataset](#dataset)
- [Models and Hyperparameters](#models-and-hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)

## Dataset

- Source: Telco Customer Churn Dataset (also available on Hugging Face/IBM).
- Size: 7,043 records, 21 columns.
- Key Features:
    - Demographics: gender, SeniorCitizen, Partner, Dependents.
    - Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.
    - Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
    - Target: Churn (Yes/No).
- Preprocessing Notes:
    - Converted TotalCharges to numeric (handle blanks as 0).
    - Mapped Churn to binary (1=Yes, 0=No).
    - One-hot encoded categorical features using scikit-learn's ColumnTransformer.
 
## Models and Hyperparameters
Three classification models are trained and tuned:

1. Logistic Regression:
    Grid: C=[0.001, 0.01, 0.1, 1, 10, 100, 1000], solver=['liblinear', 'lbfgs', ...], penalty=['l1', 'l2', ...], max_iter=[100, 200, 500].

2. K-Nearest Neighbors (KNN):
    Grid: n_neighbors=[1,3,5,...,19], weights=['uniform','distance'], metric=['euclidean',...], algorithm=['auto',...], leaf_size=[10,20,...].

3. Random Forest Classifier:
    Grid: n_estimators=[50,100,200,300,500], max_depth=[None,10,20,...], min_samples_split=[2,5,...], etc.

Hyperparameter tuning uses GridSearchCV with 5-fold CV, optimizing for accuracy (with multi-metric scoring for precision/recall).


## Evaluation Metrics

- Accuracy: Overall correctness.
- Precision: Ratio of true positives to predicted positives (minimizes false alarms).
- Recall: Ratio of true positives to actual positives (captures most churners).
- Monetary Uplift: Simulated business impact = (Predicted churners * Retention rate) * Average annual customer value.

Use scikit-learn's classification_report and custom uplift calculation.

## Results
- Best Model: Random Forest (typically ~80% accuracy on this dataset).
- Sample Scores (from notebook runs):
  - Logistic Regression: Accuracy ~0.79
  - KNN: Accuracy ~0.77
  - Random Forest: Accuracy ~0.80
- Uplift Example: ~$100,000 (assuming 20% retention success and dataset averages).

Visualizations: Confusion matrices, feature importances (via Random Forest).
