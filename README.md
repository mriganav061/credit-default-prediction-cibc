# Credit Default Prediction Model

## Overview

This project builds a machine learning model to predict the probability of credit default using borrower financial data. The goal is to support better credit risk decisions.

## Problem Statement

Predict whether a borrower will default within two years (`SeriousDlqin2yrs`).

## Dataset

* Training: ~150,000 records
* Test: ~100,000 records
* Target: SeriousDlqin2yrs

## Approach

* Missing value imputation (median)
* Model comparison:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting

## Evaluation

* Metric: ROC-AUC (handles class imbalance)

## Results

* Best Model: XGB
* ROC-AUC: ~0.86

## Key Risk Drivers

* Past due payments (30/60/90 days)
* Credit utilization
* Debt ratio
* Age

## Business Impact

* Reduces loan defaults
* Enables risk-based pricing
* Improves approval decisions

## Output

Predictions saved in:
outputs/test_probabilities.csv

## How to Run

pip install -r requirements.txt

## Future Improvements

* Add SHAP explainability
* Precision-recall tradeoff
* Threshold optimization

## Author

Mriganav Das
