# Phase 7 — ML Price Prediction

## Overview
Phase 7 trains supervised regression algorithms (Random Forest, XGBoost, and LightGBM) to accurately estimate used vehicle resale valuations based on engineered vehicle attributes.

## Model Benchmarking & Evaluation

### Evaluation Metrics Formulas
1. **Mean Absolute Error (MAE)**:
   $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

## Models Trained
- **Random Forest Regressor**: Ensemble of 150 de-correlated decision trees (`data/models/price_predictor_rf.pkl`).
- **XGBoost Regressor**: Gradient boosted decision tree ensemble with regularized loss (`data/models/price_predictor_xgb.pkl`).
- **LightGBM Regressor**: Gradient boosting framework with leaf-wise tree growth (`data/models/price_predictor_lgb.pkl`).

## Output Files
- Serialized Model Weights: `data/models/price_predictor_rf.pkl`, `data/models/price_predictor_xgb.pkl`, `data/models/price_predictor_lgb.pkl`
- Metrics Log: `data/models/model_metrics.json`, `phase_7_ml_price_prediction/evaluation_results.json`

## Execution
```bash
python phase_7_ml_price_prediction/train_models.py
```
