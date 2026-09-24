import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

def optimize_and_tune_models(feature_csv="data/processed/vehicle_features.csv", models_dir="data/models", output_dir="phase_8_model_optimization"):
    """
    Executes Phase 8 Model Optimization:
    1. Performs 5-Fold Cross-Validation across baseline models to evaluate stability.
    2. Uses GridSearchCV hyperparameter tuning on XGBoost & LightGBM to find optimal hyper-parameters.
    3. Exports tuned best model estimator and tuning summary report.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"Feature CSV '{feature_csv}' not found. Run Phase 6 first.")
        
    df = pd.read_csv(feature_csv)
    feature_cols = [c for c in df.columns if c.endswith("_encoded") or c.endswith("_scaled")]
    X = df[feature_cols]
    y = df["price"]
    
    # 1. 5-Fold Cross-Validation Evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_cv_r2 = cross_val_score(RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42), X, y, cv=kf, scoring='r2')
    xgb_cv_r2 = cross_val_score(xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42), X, y, cv=kf, scoring='r2')
    lgb_cv_r2 = cross_val_score(lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1), X, y, cv=kf, scoring='r2')
    
    cv_summary = {
        "RandomForest_5Fold_R2_Mean": round(float(np.mean(rf_cv_r2)), 4),
        "RandomForest_5Fold_R2_Std": round(float(np.std(rf_cv_r2)), 4),
        "XGBoost_5Fold_R2_Mean": round(float(np.mean(xgb_cv_r2)), 4),
        "XGBoost_5Fold_R2_Std": round(float(np.std(xgb_cv_r2)), 4),
        "LightGBM_5Fold_R2_Mean": round(float(np.mean(lgb_cv_r2)), 4),
        "LightGBM_5Fold_R2_Std": round(float(np.std(lgb_cv_r2)), 4)
    }
    print(f"[Phase 8] 5-Fold CV R² Means -> RF: {cv_summary['RandomForest_5Fold_R2_Mean']}, XGB: {cv_summary['XGBoost_5Fold_R2_Mean']}, LGB: {cv_summary['LightGBM_5Fold_R2_Mean']}")
    
    # 2. Hyperparameter Grid Search Optimization for XGBoost
    param_grid_xgb = {
        'n_estimators': [150, 250],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.03, 0.08, 0.12],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid_xgb,
        cv=3,
        scoring='r2',
        verbose=0
    )
    
    grid_search.fit(X, y)
    best_xgb = grid_search.best_estimator_
    
    best_params = grid_search.best_params_
    best_r2 = float(grid_search.best_score_)
    
    # Evaluate best model predictions
    y_preds = best_xgb.predict(X)
    best_mae = float(mean_absolute_error(y, y_preds))
    best_rmse = float(np.sqrt(mean_squared_error(y, y_preds)))
    
    # Save tuned best model
    best_model_path = os.path.join(models_dir, "tuned_price_predictor_best.pkl")
    joblib.dump(best_xgb, best_model_path)
    
    report = {
        "cross_validation_results": cv_summary,
        "grid_search_best_params": best_params,
        "best_model_metrics": {
            "R2_Score": round(best_r2, 4),
            "MAE": round(best_mae, 2),
            "RMSE": round(best_rmse, 2)
        },
        "saved_best_model_path": best_model_path
    }
    
    report_path = os.path.join(output_dir, "optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[Phase 8] Hyperparameter tuning complete. Best R²: {best_r2:.4f}, MAE: ${best_mae:,.2f}.")
    print(f"[Phase 8] Saved best tuned model to '{best_model_path}' and report to '{report_path}'.")
    return report

if __name__ == "__main__":
    optimize_and_tune_models()
