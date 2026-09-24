# Phase 8 — Model Optimization & Cross-Validation

## Overview
Phase 8 performs 5-fold cross-validation to assess generalization stability and employs Grid Search hyperparameter optimization on XGBoost & LightGBM regressors to maximize $R^2$ performance and reduce prediction variance.

## Optimization Strategy

### 1. K-Fold Cross Validation ($K=5$)
- Splits dataset into 5 disjoint stratified folds to prevent overfitting.
- Computes mean $\bar{R}^2$ and standard deviation $\sigma_{R^2}$ across all 5 evaluation iterations.

### 2. Grid Search Hyperparameter Search Space
- `n_estimators`: `[150, 250]`
- `max_depth`: `[5, 7, 9]`
- `learning_rate`: `[0.03, 0.08, 0.12]`
- `subsample`: `[0.8, 1.0]`
- `colsample_bytree`: `[0.8, 1.0]`

## Output Artefacts
- Best Tuned Model: `data/models/tuned_price_predictor_best.pkl`
- Report: `phase_8_model_optimization/optimization_report.json`

## Execution
```bash
python phase_8_model_optimization/optimize_models.py
```
